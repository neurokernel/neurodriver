
from collections import OrderedDict

import numpy as np

from neurokernel.LPU.NDComponents.MembraneModels.BaseMembraneModel import BaseMembraneModel

class MorrisLecar(BaseMembraneModel):
    params = ['V1', 'V2', 'V3', 'V4', 'phi', 'offset',
              'V_L', 'V_Ca', 'V_K', 'g_L', 'g_Ca', 'g_K']
    extra_params = ['initV', 'initn']
    internals = OrderedDict([('V', -70.0), ('n', 0.3525)])

    @property
    def maximum_dt_allowed(self):
        return 1e-5

    def pre_run(self, update_pointers):
        super(MorrisLecar, self).pre_run(update_pointers)
        self.add_initializer('initn', 'n', update_pointers)

    def get_update_template(self):
        template = """
__device__ %(internal_n)s compute_n(%(update_V)s V, %(internal_n)s n, %(param_V3)s V3, %(param_V4)s V4, %(param_phi)s phi)
{
    %(internal_n)s n_inf = 0.5 * (1 + tanh((V - V3) / V4));
    %(internal_n)s dn = phi * cosh(( V - V3) / (V4*2)) * (n_inf - n);
    return dn;
}

__device__ %(update_V)s compute_V(%(update_V)s V, %(internal_n)s n, %(input_I)s I, %(param_V1)s V1, %(param_V2)s V2,
                           %(param_offset)s offset, %(param_V_L)s V_L, %(param_V_Ca)s V_Ca,
                           %(param_V_K)s V_K, %(param_g_L)s g_L, %(param_g_K)s g_K, %(param_g_Ca)s g_Ca)
{
    %(update_V)s m_inf = 0.5 * (1+tanh((V - V1)/V2));
    %(update_V)s dV = (I - g_L * (V - V_L) - g_K * n * (V - V_K) - g_Ca * m_inf * (V - V_Ca) + offset);
    return dV;
}

__global__ void
update(int num_comps, %(dt)s dt, int nsteps,
                      %(input_I)s* g_I, %(param_V1)s* g_V1, %(param_V2)s* g_V2, %(param_V3)s* g_V3,
                      %(param_V4)s* g_V4, %(param_phi)s* g_phi, %(param_offset)s* g_offset,
                      %(param_V_L)s* g_V_L, %(param_V_Ca)s* g_V_Ca, %(param_V_K)s* g_V_K,
                      %(param_g_L)s* g_g_L, %(param_g_Ca)s* g_g_Ca, %(param_g_K)s* g_g_K,
                      %(internal_V)s* g_internalV, %(internal_n)s* g_n, %(update_V)s* g_V)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    %(update_V)s V, dV;
    %(internal_n)s n, dn;
    %(input_I)s I;
    %(param_V1)s V1;
    %(param_V2)s V2;
    %(param_V3)s V3;
    %(param_V4)s V4;
    %(param_phi)s phi;
    %(param_offset)s offset;
    %(param_V_L)s V_L;
    %(param_V_Ca)s V_Ca;
    %(param_V_K)s V_K;
    %(param_g_L)s g_L;
    %(param_g_Ca)s g_Ca;
    %(param_g_K)s g_K;

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
