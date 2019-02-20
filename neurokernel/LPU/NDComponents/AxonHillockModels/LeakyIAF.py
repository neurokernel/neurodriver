from collections import OrderedDict

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from neurokernel.LPU.NDComponents.AxonHillockModels.BaseAxonHillockModel import BaseAxonHillockModel

class LeakyIAF(BaseAxonHillockModel):
    # updates are the output states of the model
    updates = ['spike_state', # (bool)
               'V' # Membrane Potential (mV)
              ]
    # accesses are the input variables of the model
    accesses = ['I'] # (\mu A/cm^2 )
    # params are the parameters of the model that needs to be defined
    # during specification of the model
    params = ['resting_potential', # (mV)
              'threshold', # Firing Threshold (mV)
              'reset_potential', # Potential to be reset to after a spike (mV)
              'capacitance', # (\mu F/cm^2)
              'resistance' # (k\Omega cm.^2)
              ]
    # internals are the variables used to store internal states of the model,
    # and are ordered dict whose keys are the variables and value are the initial values.
    internals = OrderedDict([('internalV', 0.0)]) # Membrane Potential (mV)

    def __init__(self, params_dict, access_buffers, dt,
                 debug=False, LPU_id=None, cuda_verbose=False):
        # no need to change
        if cuda_verbose:
            self.compile_options = ['--ptxas-options=-v']
        else:
            self.compile_options = []

        self.num_comps = params_dict[params[0]].size #
        self.params_dict = params_dict
        self.access_buffers = access_buffers
        self.dt = np.double(dt)
        self.steps = 1
        self.debug = debug
        self.LPU_id = LPU_id
        self.dtype = params_dict[params[0]].dtype
        self.ddt = self.dt/self.steps

        self.internal_states = {
            c: garray.zeros(self.num_comps, dtype = self.dtype)+self.internals[c] \
            for c in self.internals}

        self.inputs = {
            k: garray.empty(self.num_comps, dtype = self.access_buffers[k].dtype)\
            for k in self.accesses}

        dtypes = {'dt': self.dtype}
        dtypes.update({k: self.inputs[k].dtype for k in self.accesses})
        dtypes.update({k: self.params_dict[k].dtype for k in self.params})
        dtypes.update({k: self.internal_states[k].dtype for k in self.internals})
        dtypes.update({k: self.dtype if not k == 'spike_state' else np.int32 for k in self.updates})
        self.update_func = self.get_update_func(dtypes)

    def pre_run(self, update_pointers):
        # copy initial value for Voltage in update and Voltage in internal state
        # change the variable names such as 'initV' or 'resting_potential'

        # if 'initV' is specified in the parameter dict,
        # it will be used as initial value
        if 'initV' in self.params_dict:
            cuda.memcpy_dtod(int(update_pointers['V']),
                             self.params_dict['initV'].gpudata,
                             self.params_dict['initV'].nbytes)
            cuda.memcpy_dtod(self.internal_states['internalV'].gpudata,
                             self.params_dict['initV'].gpudata,
                             self.params_dict['initV'].nbytes)
        else:
            # use resting potential as initial value
            cuda.memcpy_dtod(int(update_pointers['V']),
                             self.params_dict['resting_potential'].gpudata,
                             self.params_dict['resting_potential'].nbytes)
            cuda.memcpy_dtod(self.internal_states['internalV'].gpudata,
                             self.params_dict['resting_potential'].gpudata,
                             self.params_dict['resting_potential'].nbytes)

    def run_step(self, update_pointers, st=None):
        # no need to change
        for k in self.inputs:
            self.sum_in_variable(k, self.inputs[k], st=st)

        self.update_func.prepared_async_call(
            self.update_func.grid, self.update_func.block, st,
            self.num_comps, self.ddt*1000, self.steps,
            *[self.inputs[k].gpudata for k in self.accesses]+\
            [self.params_dict[k].gpudata for k in self.params]+\
            [self.internal_states[k].gpudata for k in self.internals]+\
            [update_pointers[k] for k in self.updates])

    def get_update_template(self):
        # need to update the CUDA kernel to reflect the equations of the model
        # the argument of the function must in the following order:
        # 1. int num_comps
        # 2. %(dt)s dt,
        # 3. int nsteps,
        # 4. all variables in accesses according to the order in accesses
        # 5. all variables in params according to the order in params
        # 6. all variables in internals according to the order in internals
        # 7. all variables in updates according to the order in updates
        template = """
__global__ void update(int num_comps,
               %(dt)s dt,
               int nsteps,
               %(I)s* g_I, // accesses
               %(resting_potential)s* g_resting_potential, // params
               %(threshold)s* g_threshold,
               %(reset_potential)s* g_reset_potential,
               %(capacitance)s* g_capacitance,
               %(resistance)s* g_resistance,
               %(internalV)s* g_internalV, // internals
               %(spike_state)s* g_spike_state, //updates
               %(V)s* g_V)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    // instantiate variables
    %(V)s V;
    %(I)s I;
    %(spike_state)s spike;
    %(resting_potential)s resting_potential;
    %(threshold)s threshold;
    %(reset_potential)s reset_potential;
    %(capacitance)s capacitance;
    %(resistance)s resistance;
    %(dt)s bh;

    // no need to change this for loop
    for(int i = tid; i < num_comps; i += total_threads)
    {
        // load the data from global memory
        V = g_internalV[i];
        I = g_I[i];
        capacitance = g_capacitance[i];
        resting_potential = g_resting_potential[i];
        threshold = g_threshold[i];
        resistance = g_resistance[i];
        reset_potential = g_reset_potential[i];

        // update according to equations of the model
        bh = exp%(fletter)s(-dt/(capacitance*resistance));
        V = V*bh + (resistance*I+resting_potential)*(1.0 - bh);
        spike = 0;
        if (V >= threshold)
        {
            V = reset_potential;
            spike = 1;
        }

        // write local updated states back to global memory
        g_V[i] = V;
        g_internalV[i] = V;
        g_spike_state[i] = spike;
    }
}
        """
        return template

    def get_update_func(self, dtypes):
        # no need to change
        type_dict = {k: dtype_to_ctype(dtypes[k]) for k in dtypes}
        type_dict.update({'fletter': 'f' if type_dict[params[0]] == 'float' else ''})
        mod = SourceModule(self.get_update_template() % type_dict,
                           options=self.compile_options)
        func = mod.get_function("update")
        func.prepare('i'+np.dtype(dtypes['dt']).char+'i'+'P'*(len(type_dict)-2))
        func.block = (256,1,1)
        func.grid = (min(6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT,
                         (self.num_comps-1) // 256 + 1), 1)
        return func



if __name__ == '__main__':
    import argparse
    import itertools
    import networkx as nx
    from neurokernel.tools.logging import setup_logger
    import neurokernel.core_gpu as core

    from neurokernel.LPU.LPU import LPU

    from neurokernel.LPU.InputProcessors.FileInputProcessor import FileInputProcessor
    from neurokernel.LPU.InputProcessors.StepInputProcessor import StepInputProcessor
    from neurokernel.LPU.OutputProcessors.FileOutputProcessor import FileOutputProcessor

    import neurokernel.mpi_relaunch

    dt = 1e-4
    dur = 1.0
    steps = int(dur/dt)

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=False,
                        dest='debug', action='store_true',
                        help='Write connectivity structures and inter-LPU routed data in debug folder')
    parser.add_argument('-l', '--log', default='both', type=str,
                        help='Log output to screen [file, screen, both, or none; default:none]')
    parser.add_argument('-s', '--steps', default=steps, type=int,
                        help='Number of steps [default: %s]' % steps)
    parser.add_argument('-g', '--gpu_dev', default=0, type=int,
                        help='GPU device number [default: 0]')
    args = parser.parse_args()

    file_name = None
    screen = False
    if args.log.lower() in ['file', 'both']:
        file_name = 'neurokernel.log'
    if args.log.lower() in ['screen', 'both']:
        screen = True
    logger = setup_logger(file_name=file_name, screen=screen)

    man = core.Manager()

    G = nx.MultiDiGraph()

    # specify the node
    G.add_node('neuron0', **{
               'class': 'LeakyIAF',
               'name': 'LeakyIAF',
               'resting_potential': -70.0,
               'reset_potential': -70.0,
               'threshold': -45.0,
               'capacitance': 2.0, # in mS
               'resistance': 10.0, # in Ohm
               })

    comp_dict, conns = LPU.graph_to_dicts(G)

    # use a input processor that present a step current (I) input to 'neuron0'
    # the step is from 0.2 to 0.8 and the step height is 10.0
    fl_input_processor = StepInputProcessor('I', ['neuron0'], 10.0, 0.2, 0.8)
    # output processor to record 'spike_state' and 'V' to hdf5 file 'new_output.h5',
    # with a sampling interval of 1 run step.
    fl_output_processor = FileOutputProcessor([('spike_state', None),('V', None)], 'new_output.h5', sample_interval=1)

    man.add(LPU, 'ge', dt, comp_dict, conns,
            device=args.gpu_dev, input_processors = [fl_input_processor],
            output_processors = [fl_output_processor], debug=args.debug)

    man.spawn()
    man.start(steps=args.steps)
    man.wait()

    import h5py
    import matplotlib
    matplotlib.use('PS')
    import matplotlib.pyplot as plt

    f = h5py.File('new_output.h5')
    t = np.arange(0, args.steps)*dt

    plt.figure()
    plt.plot(t,list(f['V'].values())[0])
    plt.xlabel('time, [s]')
    plt.ylabel('Voltage, [mV]')
    plt.title('Leaky Integrate-and-Fire Neuron')
    plt.savefig('lif.png',dpi=300)
