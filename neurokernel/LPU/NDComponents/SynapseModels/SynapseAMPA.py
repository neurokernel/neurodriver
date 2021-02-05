from collections import OrderedDict


import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from .BaseSynapseModel import BaseSynapseModel

# assumes a maximum of one input connection per synapse
class SynapseAMPA(BaseSynapseModel):
    accesses = ['spike_state']
    updates = ['g']
    params = ['gmax', 'st']
    internals = OrderedDict([('tst', 0.)])

    @property
    def maximum_dt_allowed(self):
        return 1e-4

    def get_update_template(self):
        template = """
__global__ void update(int num_comps, %(dt)s dt, int nsteps,
                       %(input_spike_state)s* g_spike_state,
                       %(param_gmax)s *g_gmax, %(param_st)s *g_st,
                       %(internal_tst)s *g_tst,
                       %(update_g)s *g_g)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    %(dt)s ddt = dt*1000.;
    %(input_spike_state)s spike_state;
    %(param_st)s st, est;
    %(internal_tst)s tst;

    for(int i = tid; i < num_comps; i += total_threads)
    {
        spike_state = g_spike_state[i];
        st = g_st[i];
        tst = g_tst[i];

        est = exp%(fletter)s(-ddt/st);

        for(int k = 0; k < nsteps; ++k)
        {
            tst *= est;
            if(k == 0 && (spike_state>0.0))
            {
                tst += 1;
            }
        }

        g_g[i] = tst*g_gmax[i];
        g_tst[i] = tst;
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

    G.add_node('synapse0', {
               'class': 'SynapseAMPA',
               'name': 'SynapseAMPA',
               'gmax': 0.003,#21.788,
               'st': 20.0,#100.0,
               'reverse': 0.0
               })

    comp_dict, conns = LPU.graph_to_dicts(G)

    #fl_input_processor = StepInputProcessor('I', ['neuron0'], 1, 0.2, 0.4)
    fl_input_processor = FileInputProcessor('input_spike.h5')
    fl_output_processor = FileOutputProcessor([('g', None)], 'new_output.h5', sample_interval=1)

    man.add(LPU, 'ge', dt, comp_dict, conns,
            device=args.gpu_dev, input_processors = [fl_input_processor],
            output_processors = [fl_output_processor], debug=args.debug)

    man.spawn()
    man.start(steps=args.steps)
    man.wait()
