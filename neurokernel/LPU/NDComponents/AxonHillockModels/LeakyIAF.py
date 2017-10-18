
from collections import OrderedDict

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from BaseAxonHillockModel import BaseAxonHillockModel

class LeakyIAF(BaseAxonHillockModel):
    updates = ['spike_state', 'V']
    accesses = ['I']
    states = OrderedDict([('V', -65.)])
    params = OrderedDict([
        ('resting_potential', 0.),
        ('threshold', -25.),
        ('reset_potential', -65.),
        ('capacitance', 0.065),
        ('resistance', 1000.)])
    max_dt = 1e-4
    cuda_src = """
# if (defined(USE_DOUBLE))
#    define FLOATTYPE double
#    define EXP exp
#    define POW pow
# else
#    define FLOATTYPE float
#    define EXP expf
#    define POW powf
# endif
#
# if (defined(USE_LONG_LONG))
#     define INTTYPE long long
# else
#     define INTTYPE int
# endif
#
__global__ void update(
    int num_comps,
    FLOATTYPE dt, int nsteps,
    FLOATTYPE *g_I,
    FLOATTYPE *g_resting_potential,
    FLOATTYPE *g_threshold,
    FLOATTYPE *g_reset_potential,
    FLOATTYPE *g_capacitance,
    FLOATTYPE *g_resistance,
    FLOATTYPE *g_internalV,
    INTTYPE *g_spike_state,
    FLOATTYPE *g_V)
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
    FLOATTYPE resistance;
    FLOATTYPE bh;

    for(int i = tid; i < num_comps; i += total_threads)
    {
        V = g_internalV[i];
        I = g_I[i];
        capacitance = g_capacitance[i];
        resting_potential = g_resting_potential[i];
        threshold = g_threshold[i];
        resistance = g_resistance[i];
        reset_potential = g_reset_potential[i];

        bh = EXP(-dt/(capacitance*resistance));
        V = V*bh + (resistance*I+resting_potential)*(1.0 - bh);
        spike = 0;
        if (V >= threshold) {
            V = reset_potential;
            spike = 1;
        }

        g_V[i] = V;
        g_internalV[i] = V;
        g_spike_state[i] = spike;
    }
}
"""

    def run_step(self, update_pointers, st=None):
        for k in self.inputs:
            self.sum_in_variable(k, self.inputs[k], st=st)

        self.update_func.prepared_async_call(
            self.update_func.grid, self.update_func.block, st,
            self.num_comps, 1000.*self.dt, self.steps,
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
               'class': 'LeakyIAF',
               'name': 'LeakyIAF',
               'resting_potential': -70.0,
               'threshold': -45.0,
               'capacitance': 0.07, # in mS
               'resistance': 0.2, # in Ohm
               })

    comp_dict, conns = LPU.graph_to_dicts(G)

    fl_input_processor = StepInputProcessor('I', ['neuron0'], 40, 0.2, 0.8)
    fl_output_processor = FileOutputProcessor([('spike_state', None),('V', None)], 'new_output.h5', sample_interval=1)

    man.add(LPU, 'ge', dt, comp_dict, conns,
            device=args.gpu_dev, input_processors = [fl_input_processor],
            output_processors = [fl_output_processor], debug=args.debug)

    man.spawn()
    man.start(steps=args.steps)
    man.wait()
