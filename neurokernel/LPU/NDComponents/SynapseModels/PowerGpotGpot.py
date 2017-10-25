
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
    params = OrderedDict([
        ('threshold', -55.0),
        ('slope', 0.02),
        ('power', 1.0),
        ('saturation', 0.4),
        ('reverse', -50.)])
    states = OrderedDict()
    max_dt = None
    cuda_src = """
# if (defined(USE_DOUBLE))
#    define FLOATTYPE double
#    define EXP exp
#    define POW pow
#    define FMAX fmax
#    define FMIN fmin
# else
#    define FLOATTYPE float
#    define EXP expf
#    define POW powf
#    define FMAX fmaxf
#    define FMIN fminf
# endif
#
# if (defined(USE_LONG_LONG))
#     define INTTYPE long long
# else
#     define INTTYPE int
# endif

__global__ void PowerGPotGPot(int num_comps, FLOATTYPE dt, int steps,
    FLOATTYPE *g_V,
    FLOATTYPE *g_threshold,
    FLOATTYPE *g_slope,
    FLOATTYPE *g_power,
    FLOATTYPE *g_saturation,
    FLOATTYPE *g_g)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    FLOATTYPE V;
    FLOATTYPE threshold;
    FLOATTYPE slope;
    FLOATTYPE power;
    FLOATTYPE saturation;

    for (int i = tid; i < num_comps; i += total_threads) {
        V = g_V[i];
        threshold = g_threshold[i];
        slope = g_slope[i];
        power = g_power[i];
        saturation = g_saturation[i];

        g_g[i] = FMIN(saturation, slope*POW(fmax(0.0,V-threshold),power));
    }
}
"""
    def __init__(self, params_dict, access_buffers, dt, LPU_id=None,
        debug=False, cuda_verbose=False):
        super(PowerGPotGPot, self).__init__(params_dict, access_buffers, dt,
            LPU_id=LPU_id, debug=debug, cuda_verbose=cuda_verbose)

        self.retrieve_buffer_funcs = {}
        for k in self.accesses:
            self.retrieve_buffer_funcs[k] = \
                self.get_retrieve_buffer_func(
                    k, dtype = self.access_buffers[k].dtype)

    def run_step(self, update_pointers, st = None):
        # retrieve all buffers into a linear array
        for k in self.inputs:
            self.retrieve_buffer(k, st = st)

        self.update_func.prepared_async_call(
            self.update_func.grid, self.update_func.block, st,
            self.num_comps, self.dt*1000, self.steps,
            *[self.inputs[k].gpudata for k in self.accesses]+\
            [self.params_dict[k].gpudata for k in self.params if k != 'reverse']+\
            [self.states[k].gpudata for k in self.states]+\
            [update_pointers[k] for k in self.updates])

    def get_update_func(self):
        mod = SourceModule(self.cuda_src, options=self.compile_options)
        func = mod.get_function("PowerGPotGPot")
        func.prepare('i'+np.dtype(self.floattype).char+'i'+'P'*(self.num_garray-1))
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

    # plot the result
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    f = h5py.File('new_output.h5')
    t = np.arange(0, args.steps)*dt
    V = np.linspace(-70., -30., len(t))

    plt.figure()
    plt.subplot(211)
    plt.plot(t, V)
    plt.title('Pre-Synaptic Neuron Voltage')
    plt.xlabel('time, [s]')
    plt.ylabel('Voltage, [mV]')
    plt.xlim([0, dur])
    plt.grid()

    plt.subplot(212)
    plt.plot(t,f['g'].values()[0])
    plt.xlabel('time, [s]')
    plt.ylabel('Conductance, [mS]')
    plt.title('Power Graded Potential-Graded Potential Synapse')
    plt.xlim([0, dur])
    plt.grid()
    plt.tight_layout()
    plt.savefig('power_gpot_gpot.png', dpi=300)
