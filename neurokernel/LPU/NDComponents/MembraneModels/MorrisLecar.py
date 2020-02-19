
from collections import OrderedDict

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from neurokernel.LPU.NDComponents.MembraneModels.BaseMembraneModel import BaseMembraneModel

class MorrisLecar(BaseMembraneModel):
    params = ['V1', 'V2', 'V3', 'V4', 'phi', 'offset',
              'V_L', 'V_Ca', 'V_K', 'g_L', 'g_Ca', 'g_K']
    internals = OrderedDict([('internalV', -70.0), ('n', 0.3525)])

    def __init__(self, params_dict, access_buffers, dt, LPU_id=None,
                 debug=False, cuda_verbose=False):
        if cuda_verbose:
            self.compile_options = ['--ptxas-options=-v']
        else:
            self.compile_options = []

        self.num_comps = params_dict[self.params[0]].size
        self.params_dict = params_dict
        self.access_buffers = access_buffers
        self.dt = np.double(dt)
        self.steps = max(int(round(dt / 1e-5)), 1)
        self.debug = debug
        self.LPU_id = LPU_id
        self.dtype = params_dict[self.params[0]].dtype
        self.ddt = dt / self.steps

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
        dtypes.update({k: self.dtype for k in self.updates})
        self.update_func = self.get_update_func(dtypes)

    def pre_run(self, update_pointers):
        #initializing
        cuda.memcpy_dtod(int(update_pointers['V']),
                         self.params_dict['initV'].gpudata,
                         self.params_dict['initV'].nbytes)
        cuda.memcpy_dtod(self.internal_states['internalV'].gpudata,
                         self.params_dict['initV'].gpudata,
                         self.params_dict['initV'].nbytes)
        cuda.memcpy_dtod(self.internal_states['n'].gpudata,
                         self.params_dict['initn'].gpudata,
                         self.params_dict['initn'].nbytes)


    def run_step(self, update_pointers, st=None):
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
        template = """
__device__ %(n)s compute_n(%(V)s V, %(n)s n, %(V3)s V3, %(V4)s V4, %(phi)s phi)
{
    %(n)s n_inf = 0.5 * (1 + tanh((V - V3) / V4));
    %(n)s dn = phi * cosh(( V - V3) / (V4*2)) * (n_inf - n);
    return dn;
}

__device__ %(V)s compute_V(%(V)s V, %(n)s n, %(I)s I, %(V1)s V1, %(V2)s V2,
                           %(offset)s offset, %(V_L)s V_L, %(V_Ca)s V_Ca,
                           %(V_K)s V_K, %(g_L)s g_L, %(g_K)s g_K, %(g_Ca)s g_Ca)
{
    %(V)s m_inf = 0.5 * (1+tanh((V - V1)/V2));
    %(V)s dV = (I - g_L * (V - V_L) - g_K * n * (V - V_K) - g_Ca * m_inf * (V - V_Ca) + offset);
    return dV;
}

__global__ void
morris_lecar_multiple(int num_comps, %(dt)s dt, int nsteps,
                      %(I)s* g_I, %(V1)s* g_V1, %(V2)s* g_V2, %(V3)s* g_V3,
                      %(V4)s* g_V4, %(phi)s* g_phi, %(offset)s* g_offset,
                      %(V_L)s* g_V_L, %(V_Ca)s* g_V_Ca, %(V_K)s* g_V_K,
                      %(g_L)s* g_g_L, %(g_Ca)s* g_g_Ca, %(g_K)s* g_g_K,
                      %(internalV)s* g_internalV, %(n)s* g_n, %(V)s* g_V)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    %(V)s V, dV;
    %(n)s n, dn;
    %(I)s I;
    %(V1)s V1;
    %(V2)s V2;
    %(V3)s V3;
    %(V4)s V4;
    %(phi)s phi;
    %(offset)s offset;
    %(V_L)s V_L;
    %(V_Ca)s V_Ca;
    %(V_K)s V_K;
    %(g_L)s g_L;
    %(g_Ca)s g_Ca;
    %(g_K)s g_K;

    for(int k = tid; k < num_comps; k += total_threads)
    {
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

        for(int i = 0; i < nsteps; ++i)
        {
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
        return template

    def get_update_func(self, dtypes):
        type_dict = {k: dtype_to_ctype(dtypes[k]) for k in dtypes}
        type_dict.update({'fletter': 'f' if type_dict[self.params[0]] == 'float' else ''})
        mod = SourceModule(self.get_update_template() % type_dict,
                           options=self.compile_options)
        func = mod.get_function("morris_lecar_multiple")
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

    fl_input_processor = StepInputProcessor('I', ['neuron0'], 5, 0.2, 0.4)
    fl_output_processor = FileOutputProcessor([('V', None)], 'new_output.h5', sample_interval=1)

    man.add(LPU, 'ge', dt, comp_dict, conns,
            device=args.gpu_dev, input_processors = [fl_input_processor],
            output_processors = [fl_output_processor], debug=args.debug)

    man.spawn()
    man.start(steps=args.steps)
    man.wait()
