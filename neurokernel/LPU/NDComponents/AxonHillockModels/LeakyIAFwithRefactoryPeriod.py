
from collections import OrderedDict

import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from neurokernel.LPU.utils.simpleio import *
from BaseAxonHillockModel import BaseAxonHillockModel

class LeakyIAFwithRefactoryPeriod(BaseAxonHillockModel):
    updates = ['spike_state', 'V']
    accesses = ['I']
    states = OrderedDict([('V', -65.)])
    params = OrderedDict([
        ('resting_potential', 0.),
        ('threshold', -25.),
        ('reset_potential', -65.),
        ('capacitance', 0.065),
        ('refractory_period', 0.0),
        ('time_constant', 16.0),
        ('bias_current', 0.0)])
    max_dt = 1e-4

    cuda_src = """
# if (defined(USE_DOUBLE))
#    define FLOATTYPE double
#    define EXP exp
#    define POW pow
#    define FMAX fmax
# else
#    define FLOATTYPE float
#    define EXP expf
#    define POW powf
#    define FMAX fmaxf
# endif
#
# if (defined(USE_LONG_LONG))
#     define INTTYPE long long
# else
#     define INTTYPE int
# endif
#
__global__ void update(int num_comps, FLOATTYPE dt, INTTYPE steps,
           FLOATTYPE* g_I,
           FLOATTYPE* g_resting_potential,
           FLOATTYPE* g_threshold,
           FLOATTYPE* g_reset_potential,
           FLOATTYPE* g_capacitance,
           FLOATTYPE* g_refractory_period,
           FLOATTYPE* g_time_constant,
           FLOATTYPE* g_bias_current,
           FLOATTYPE* g_refractory_time_left,
           INTTYPE* g_spike_state, FLOATTYPE* g_V)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    FLOATTYPE V;
    FLOATTYPE I;
    INTTYPE spike;
    FLOATTYPE resting_potential;
    FLOATTYPE threshold;
    FLOATTYPE reset_potential;
    FLOATTYPE capacitance;
    FLOATTYPE time_constant;
    FLOATTYPE bias_current;
    FLOATTYPE refractory_time_left;
    FLOATTYPE bh;

    for (int i = tid; i < num_comps; i += total_threads)
    {
        refractory_time_left = FMAX(g_refractory_time_left[i] - dt, 0);

        V = g_V[i];
        I = g_I[i];
        time_constant = g_time_constant[i];
        capacitance = g_capacitance[i];
        reset_potential = g_reset_potential[i];
        resting_potential = g_resting_potential[i];
        threshold = g_threshold[i];
        bias_current = g_bias_current[i];

        bh = EXP(-dt/time_constant);
        V = V*bh + ((refractory_time_left == 0 ? time_constant/capacitance*(I+bias_current) : 0) + resting_potential) * (1.0 - bh);

        spike = 0;
        if (V >= threshold)
        {
            V = reset_potential;
            spike = 1;
            refractory_time_left += g_refractory_period[i];
        }

        g_V[i] = V;
        g_spike_state[i] = spike;
        g_refractory_time_left[i] = refractory_time_left;

    }
}
    """

    def run_step(self, update_pointers, st=None):
        for k in self.inputs:
            self.sum_in_variable(k, self.inputs[k], st=st)

        self.update_func.prepared_async_call(
            self.update_func.grid, self.update_func.block, st,
            self.num_comps, self.dt*1000, self.steps,
            *[self.inputs[k].gpudata for k in self.accesses]+\
            [self.params_dict[k].gpudata for k in self.params]+\
            [self.states[k].gpudata for k in self.states]+\
            [update_pointers[k] for k in self.updates])

    def get_update_func(self):
        mod = SourceModule(self.cuda_src, options=self.compile_options)
        func = mod.get_function("update")
        func.prepare('i'+np.dtype(self.floattype).char+'i'+'P'*self.num_garray)
        func.block = (256,1,1)
        func.grid = (min(6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT,
                         (self.num_comps-1) / 256 + 1), 1)
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
    parser.add_argument('-l', '--log', default='none', type=str,
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

    G.add_node('neuron0', **{
               'class': 'LeakyIAFwithRefactoryPeriod',
               'name': 'LIF',
               'resting_potential': -70.0,
               'threshold': -45.0,
               'reset_potential': -55.0,
               'capacitance': 0.0744005237682, # in mS
               'refractory_period': 0.0, # in milliseconds
               'time_constant': 16.0, # in milliseconds
               'bias_current': 0.0
               })

    comp_dict, conns = LPU.graph_to_dicts(G)

    fl_input_processor = StepInputProcessor('I', ['neuron0'], 1, 0.2, 0.4)
    fl_output_processor = FileOutputProcessor([('spike_state', None),('V', None)], 'new_output.h5', sample_interval=1)

    man.add(LPU, 'ge', dt, comp_dict, conns, cuda_verbose=True,
            device=args.gpu_dev, input_processors = [fl_input_processor],
            output_processors = [fl_output_processor], debug=args.debug)

    print args.steps
    man.spawn()
    man.start(steps=args.steps)
    man.wait()

    # plot the result
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    f = h5py.File('new_output.h5')
    t = np.arange(0, args.steps)*dt

    plt.figure()
    plt.subplot(211)
    plt.plot(t,f['V'].values()[0])
    plt.xlabel('time, [s]')
    plt.ylabel('Voltage, [mV]')
    plt.title('LIF Neuron')
    plt.xlim([0, dur])
    #plt.ylim([-70, 60])
    plt.grid()
    plt.subplot(212)
    spk = f['spike_state/data'].value.flatten().nonzero()[0]
    plt.stem(t[spk],np.ones((len(spk),)))
    plt.xlabel('time, [s]')
    plt.ylabel('Spike')
    plt.xlim([0, dur])
    plt.ylim([0, 1.2])
    plt.grid()
    plt.tight_layout()
    plt.savefig('lif.png',dpi=300)
