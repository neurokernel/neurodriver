
from collections import OrderedDict

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from BaseSynapseModel import BaseSynapseModel

#This class assumes a single pre synaptic connection per component instance
class PowerGPotGPot(BaseSynapseModel):
    accesses = ['V']
    updates = ['g']
    params = ['threshold', 'slope', 'power', 'saturation']
    internals = OrderedDict([])

    def __init__(self, params_dict, access_buffers, dt,
                 LPU_id=None, debug=False, cuda_verbose=False):
        if cuda_verbose:
            self.compile_options = ['--ptxas-options=-v']
        else:
            self.compile_options = []

        self.debug = debug
        self.dt = dt
        self.num_comps = params_dict['threshold'].size
        self.dtype = params_dict['threshold'].dtype
        self.LPU_id = LPU_id
        self.nsteps = 1
        self.ddt = dt/self.nsteps

        self.params_dict = params_dict
        self.access_buffers = access_buffers

        self.internal_states = {
            c: garray.zeros(self.num_comps, dtype = self.dtype)+self.internals[c] \
            for c in self.internals}

        self.inputs = {
            k: garray.empty(self.num_comps, dtype = self.access_buffers[k].dtype)\
            for k in self.accesses}

        self.retrieve_buffer_funcs = {}
        for k in self.accesses:
            self.retrieve_buffer_funcs[k] = \
                self.get_retrieve_buffer_func(
                    k, dtype = self.access_buffers[k].dtype)

        dtypes = {'dt': self.dtype}
        dtypes.update({k: self.inputs[k].dtype for k in self.accesses})
        dtypes.update({k: self.params_dict[k].dtype for k in self.params})
        dtypes.update({k: self.internal_states[k].dtype for k in self.internals})
        dtypes.update({k: self.dtype if not k == 'spike_state' else np.int32 for k in self.updates})
        self.update_func = self.get_update_func(dtypes)

    def run_step(self, update_pointers, st = None):
        # retrieve all buffers into a linear array
        for k in self.inputs:
            self.retrieve_buffer(k, st = st)

        self.update_func.prepared_async_call(
            self.update_func.grid, self.update_func.block, st,
            self.num_comps, self.ddt*1000, self.nsteps,
            *[self.inputs[k].gpudata for k in self.accesses]+\
            [self.params_dict[k].gpudata for k in self.params]+\
            [self.internal_states[k].gpudata for k in self.internals]+\
            [update_pointers[k] for k in self.updates])

    def get_update_template(self):
        template = """
__global__ void PowerGPotGPot(int num_comps, %(dt)s dt, int steps,
                       %(V)s* g_V, %(threshold)s* g_threshold,
                       %(slope)s* g_slope, %(power)s* g_power,
                       %(saturation)s* g_saturation,
                       %(g)s* g_g)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    %(V)s V;
    %(threshold)s threshold;
    %(slope)s slope;
    %(power)s power;
    %(saturation)s saturation;

    for(int i = tid; i < num_comps; i += total_threads)
    {
        V = g_V[i];
        threshold = g_threshold[i];
        slope = g_slope[i];
        power = g_power[i];
        saturation = g_saturation[i];

        g_g[i] = fmin%(fletter)s(saturation,
                    slope*pow%(fletter)s(fmax(0.0,V-threshold),power));
    }
}
"""
        return template

    def get_update_func(self, dtypes):
        type_dict = {k: dtype_to_ctype(dtypes[k]) for k in dtypes}
        type_dict.update({'fletter': 'f' if type_dict['threshold'] == 'float' else ''})
        mod = SourceModule(self.get_update_template() % type_dict,
                           options=self.compile_options)
        func = mod.get_function("PowerGPotGPot")
        func.prepare('i'+np.dtype(dtypes['dt']).char+'i'+'P'*(len(type_dict)-2))
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

    from neurokernel.LPU.InputProcessors.RampInputProcessor import RampInputProcessor
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

    G.add_node('synapse0', **{
               'class': 'PowerGPotGPot',
               'name': 'PowerGPotGPot',
               'gmax': 0.4,
               'threshold': -55.0,
               'slope': 0.02,
               'power': 1.0,
               'saturation': 0.4,
               'reverse': 0.0
               })

    comp_dict, conns = LPU.graph_to_dicts(G)

    fl_input_processor = RampInputProcessor('V', ['synapse0'], 0.0, 1.0, -70.0, -30.0)
    fl_output_processor = FileOutputProcessor([('g', None)], 'new_output.h5', sample_interval=1)

    man.add(LPU, 'ge', dt, comp_dict, conns,
            device=args.gpu_dev, input_processors = [fl_input_processor],
            output_processors = [fl_output_processor], debug=args.debug)

    man.spawn()
    man.start(steps=args.steps)
    man.wait()
