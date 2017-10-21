
from collections import OrderedDict

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from BaseMembraneModel import BaseMembraneModel

class MorrisLecar(BaseMembraneModel):
    params = OrderedDict([
        ('V1', 30.), ('V2', 15.), ('V3', 0.), ('V4', 30.), ('phi', 0.025),
        ('offset', 0.), ('V_L', -50.), ('V_Ca', 100.0), ('V_K', -70.0),
        ('g_L', 0.5), ('g_Ca', 1.1), ('g_K', 2.0)])
    states = OrderedDict([('V', -70.), ('n', 0.3525)])
    max_dt = 1e-5
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

__device__ FLOATTYPE compute_n(
    FLOATTYPE V, FLOATTYPE n, FLOATTYPE V3, FLOATTYPE V4, FLOATTYPE phi)
{
    FLOATTYPE n_inf = 0.5 * (1 + tanh((V - V3) / V4));
    FLOATTYPE dn = phi * cosh(( V - V3) / (V4*2)) * (n_inf - n);
    return dn;
}

__device__ FLOATTYPE compute_V(
    FLOATTYPE V, FLOATTYPE n, FLOATTYPE I, FLOATTYPE V1, FLOATTYPE V2,
    FLOATTYPE offset, FLOATTYPE V_L, FLOATTYPE V_Ca,
    FLOATTYPE V_K, FLOATTYPE g_L, FLOATTYPE g_K, FLOATTYPE g_Ca)
{
    FLOATTYPE m_inf = 0.5 * (1+tanh((V - V1)/V2));
    FLOATTYPE dV = (I - g_L * (V - V_L) - g_K * n * (V - V_K)
        - g_Ca * m_inf * (V - V_Ca) + offset);
    return dV;
}

__global__ void
morris_lecar_multiple(
    INTTYPE num_comps, FLOATTYPE dt, INTTYPE nsteps,
    FLOATTYPE* g_I, FLOATTYPE* g_V1, FLOATTYPE* g_V2, FLOATTYPE* g_V3,
    FLOATTYPE* g_V4, FLOATTYPE* g_phi, FLOATTYPE* g_offset,
    FLOATTYPE* g_V_L, FLOATTYPE* g_V_Ca, FLOATTYPE* g_V_K,
    FLOATTYPE* g_g_L, FLOATTYPE* g_g_Ca, FLOATTYPE* g_g_K,
    FLOATTYPE* g_internalV, FLOATTYPE* g_n, FLOATTYPE* g_V)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    FLOATTYPE V, dV;
    FLOATTYPE n, dn;
    FLOATTYPE I;
    FLOATTYPE V1;
    FLOATTYPE V2;
    FLOATTYPE V3;
    FLOATTYPE V4;
    FLOATTYPE phi;
    FLOATTYPE offset;
    FLOATTYPE V_L;
    FLOATTYPE V_Ca;
    FLOATTYPE V_K;
    FLOATTYPE g_L;
    FLOATTYPE g_Ca;
    FLOATTYPE g_K;

    for (int k = tid; k < num_comps; k += total_threads) {
        V = g_internalV[k];
        n = g_n[k];
        V1 = g_V1[k];
        V2 = g_V2[k];
        V3 = g_V3[k];
        V4 = g_V4[k];
        phi = g_phi[k];
        offset = g_offset[k];
        V_L = g_V_L[k];
        V_Ca = g_V_Ca[k];
        V_K = g_V_K[k];
        g_L = g_g_L[k];
        g_Ca = g_g_Ca[k];
        g_K = g_g_K[k];
        I = g_I[k];

        for (int i = 0; i < nsteps; ++i) {
            dn = compute_n(V, n, V3, V4, phi);
            dV = compute_V(V, n, I, V1, V2,
                           offset, V_L, V_Ca, V_K,
                           g_L, g_K, g_Ca);
            V += dV * dt;
            n += dn * dt;
        }

        g_V[k] = V;
        g_internalV[k] = V;
        g_n[k] = n;
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
        func = mod.get_function("morris_lecar_multiple")
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
               'class': 'MorrisLecar',
               'name': 'MorrisLecar',
               'V1': -20.,
               'V2': 50.,
               'V3': -40.,
               'V4': 20.,
               'phi': 0.001,
               'offset': 0.,
               'V_L': -40.,
               'V_Ca': 120.,
               'V_K': -80.,
               'g_L': 3.,
               'g_Ca': 4.,
               'g_K': 16.,
               'initV': -46.080,
               'initn': 0.3525
               })

    comp_dict, conns = LPU.graph_to_dicts(G)

    fl_input_processor = StepInputProcessor('I', ['neuron0'], 10, 0.2, 0.4)
    fl_output_processor = FileOutputProcessor([('V', None)], 'new_output.h5', sample_interval=1)

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

    plt.figure()
    plt.subplot(111)
    plt.plot(t,f['V'].values()[0])
    plt.xlabel('time, [s]')
    plt.ylabel('Voltage, [mV]')
    plt.title('Morris-Lecar Neuron')
    plt.xlim([0, dur])
    plt.ylim([-50, -30])
    plt.grid()
    plt.tight_layout()
    plt.savefig('mln.png',dpi=300)
