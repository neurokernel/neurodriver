
from collections import OrderedDict

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from neurokernel.LPU.NDComponents.SynapseModels.BaseSynapseModel import BaseSynapseModel

class AlphaSynapse(BaseSynapseModel):
    accesses = ['spike_state'] # (bool)
    updates = ['g'] # conductance (mS/cm^2)
    params = ['gmax', # maximum conductance (mS/cm^2)
              'ar', # rise rate of conductance (ms)
              'ad' # decay rate of conductance (ms)
              ]
    internals = OrderedDict([('z', 0.0),  # g,
                             ('dz', 0.0),  # derivative of g
                             ('d2z', 0.0)  # second derivative of g
                             ])

    def __init__(self, params_dict, access_buffers, dt,
                 LPU_id=None, debug=False, cuda_verbose=False):
        if cuda_verbose:
            self.compile_options = ['--ptxas-options=-v']
        else:
            self.compile_options = []

        self.debug = debug
        self.dt = dt
        self.num_comps = params_dict[self.params[0]].size
        self.dtype = params_dict[self.params[0]].dtype
        self.LPU_id = LPU_id
        self.nsteps = 1
        self.ddt = dt / self.nsteps

        self.params_dict = params_dict
        self.access_buffers = access_buffers

        self.internal_states = {
            c: garray.zeros(self.num_comps, dtype = self.dtype) + self.internals[c]
            for c in self.internals}

        self.inputs = {
            k: garray.empty(self.num_comps, dtype=self.access_buffers[k].dtype)
            for k in self.accesses}

        self.retrieve_buffer_funcs = {}
        for k in self.accesses:
            self.retrieve_buffer_funcs[k] = \
                self.get_retrieve_buffer_func(
                    k, dtype=self.access_buffers[k].dtype)

        dtypes = {'dt': self.dtype}
        dtypes.update({k.format(k): self.inputs[
                      k].dtype for k in self.accesses})
        dtypes.update({k: self.params_dict[
                      k].dtype for k in self.params})
        dtypes.update({k: self.internal_states[
                      k].dtype for k in self.internals})
        dtypes.update({k: self.dtype if not k ==
                       'spike_state' else np.int32 for k in self.updates})
        self.update_func = self.get_update_func(dtypes)

    def run_step(self, update_pointers, st=None):
        # retrieve all buffers into a linear array
        for k in self.inputs:
            self.retrieve_buffer(k, st=st)

        self.update_func.prepared_async_call(
            self.update_func.grid, self.update_func.block, st,
            self.num_comps, self.ddt, self.nsteps,
            *[self.inputs[k].gpudata for k in self.accesses] +
            [self.params_dict[k].gpudata for k in self.params] +
            [self.internal_states[k].gpudata for k in self.internals] +
            [update_pointers[k] for k in self.updates])

    def get_update_template(self):
        # The following kernel assumes a maximum of one input connection
        # per neuron
        if self.nsteps == 1:
            # this is a kernel that runs 1 step internally for each self.dt
            template = """
__global__ void alpha_synapse(int num_comps, %(dt)s dt, int steps,
                       %(spike_state)s* g_spike_state,
                       %(gmax)s* g_gmax, %(ar)s* g_ar,
                       %(ad)s* g_ad,
                       %(z)s* g_z, %(dz)s* g_dz,
                       %(d2z)s* g_d2z, %(g)s* g_g)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    %(spike_state)s spike_state;
    %(gmax)s gmax;
    %(ar)s ar;
    %(ad)s ad;
    %(z)s z, new_z;
    %(dz)s dz, new_dz;
    %(d2z)s d2z, new_d2z;

    for(int i = tid; i < num_comps; i += total_threads)
    {
        ar = g_ar[i];
        ad = g_ad[i];
        z = g_z[i];
        dz = g_dz[i];
        d2z = g_d2z[i];
        spike_state = g_spike_state[i];

        new_z = fmax( 0., z + dt*dz );
        new_dz = dz + dt*d2z;
        if( spike_state )
            new_dz += ar*ad;
        new_d2z = -( ar+ad )*dz - ar*ad*z;

        gmax = g_gmax[i];
        g_z[i] = new_z;
        g_dz[i] = new_dz;
        g_d2z[i] = new_d2z;
        g_g[i] = new_z*gmax;
    }
}
"""
        else:
            # this is a kernel that runs self.nstep steps internally for each self.dt
            # see the "k" for loop
            template = """
__global__ void update(int num_comps, %(dt)s dt, int steps,
                       %(spike_state)s* g_spike_state,
                       %(gmax)s* g_gmax, %(ar)s* g_ar,
                       %(ad)s* g_ad,
                       %(z)s* g_z, %(dz)s* g_dz,
                       %(d2z)s* g_d2z, %(g)s* g_g)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    %(spike_state)s spike_state;
    %(gmax)s gmax;
    %(ar)s ar;
    %(ad)s ad;
    %(z)s z, new_z;
    %(dz)s dz, new_dz;
    %(d2z)s d2z, new_d2z;

    for(int i = tid; i < num_comps; i += total_threads)
    {
        ar = g_ar[i];
        ad = g_ad[i];
        z = g_z[i];
        dz = g_dz[i];
        d2z = g_d2z[i];
        spike_state = g_spike_state[i];

        for(int k = 0; k < nsteps; ++k)
        {
            new_z = fmax( 0., z + dt*dz );
            new_dz = dz + dt*d2z;
            if(k == 0 && spike_state)
                new_dz += ar*ad;
            new_d2z = -( ar+ad )*dz - ar*ad*z;

            z = new_z;
            dz = new_dz;
            d2z = new_d2z;
        }

        gmax = g_gmax[i];
        g_z[i] = new_z;
        g_dz[i] = new_dz;
        g_d2z[i] = new_d2z;
        g_g[i] = new_z*gmax;
    }
}
"""
        return template

    def get_update_func(self, dtypes):
        type_dict = {k: dtype_to_ctype(dtypes[k]) for k in dtypes}
        type_dict.update({'fletter': 'f' if type_dict[self.params[0]] == 'float' else ''})
        mod = SourceModule(self.get_update_template() % type_dict,
                           options=self.compile_options)
        func = mod.get_function("update")
        func.prepare(
            'i' + np.dtype(dtypes['dt']).char + 'i' + 'P' * (len(type_dict) - 2))
        func.block = (256, 1, 1)
        func.grid = (min(6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT,
                         (self.num_comps - 1) / 256 + 1), 1)
        return func


if __name__ == '__main__':
    import argparse
    import itertools

    import networkx as nx
    import h5py

    from neurokernel.tools.logging import setup_logger
    import neurokernel.core_gpu as core
    from neurokernel.LPU.LPU import LPU
    from neurokernel.LPU.InputProcessors.FileInputProcessor import FileInputProcessor
    from neurokernel.LPU.OutputProcessors.FileOutputProcessor import FileOutputProcessor
    import neurokernel.mpi_relaunch

    dt = 1e-6
    dur = 1.0
    steps = int(dur / dt)

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

    t = np.arange(0, dt * steps, dt)

    uids = np.array(["synapse0"], dtype='S')

    spike_state = np.zeros((steps, 1), dtype=np.int32)
    spike_state[np.nonzero(t - np.round(t / 0.04) * 0.04 == 0)[0]] = 1

    with h5py.File('input_spike.h5', 'w') as f:
        f.create_dataset('spike_state/uids', data=uids)
        f.create_dataset('spike_state/data', (steps, 1),
                         dtype=np.int32,
                         data=spike_state)

    man = core.Manager()

    G = nx.MultiDiGraph()

    G.add_node('synapse0', **{
               'class': 'AlphaSynapse',
               'name': 'AlphaSynapse',
               'gmax': 0.003 * 1e-3,
               'ar': 110.0,
               'ad': 190.0,
               'reverse': 0.0
               })

    comp_dict, conns = LPU.graph_to_dicts(G)

    fl_input_processor = FileInputProcessor('input_spike.h5')
    fl_output_processor = FileOutputProcessor(
        [('g', None)], 'new_output.h5', sample_interval=1)

    man.add(LPU, 'ge', dt, comp_dict, conns,
            device=args.gpu_dev, input_processors=[fl_input_processor],
            output_processors=[fl_output_processor], debug=args.debug)

    man.spawn()
    man.start(steps=args.steps)
    man.wait()
