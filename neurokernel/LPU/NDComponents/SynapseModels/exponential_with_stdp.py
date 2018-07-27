
from collections import OrderedDict

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from BaseSynapseModel import BaseSynapseModel

#This class assumes a single pre synaptic connection per component instance
class exponential_with_stdp(BaseSynapseModel):
    accesses = ['spike_state']
    access_inputs = [2]
    updates = ['g']
    params = ['tau_g', # time constant for exponential decay of conductance
              'tau_plus', # time constant for pre-before-post rule
              'tau_minus', # time constant for post-before-pre rule
              'eta_plus', 'eta_minus', # soft bounds
              'w_max', 'w_min'] # max and min weights
    internals = OrderedDict([('internal_g', 0.0), ('weight', 0.5), ('x', 0.0), ('y', 0.0)])

    def __init__(self, params_dict, access_buffers, dt,
                 LPU_id=None, debug=False, cuda_verbose=False):
        if cuda_verbose:
            self.compile_options = ['--ptxas-options=-v']
        else:
            self.compile_options = []

        self.debug = debug
        self.dt = dt
        self.num_comps = params_dict['tau_g'].size
        self.dtype = params_dict['tau_g'].dtype
        self.LPU_id = LPU_id
        self.nsteps = 1
        self.ddt = dt/self.nsteps

        self.params_dict = params_dict
        self.access_buffers = access_buffers

        self.internal_states = {
            c: garray.zeros(self.num_comps, dtype = self.dtype)+self.internals[c] \
            for c in self.internals}

        self.N_inputs = {k: self.access_inputs[i] for i, k in enumerate(self.accesses)}
        self.inputs = {
            k: garray.empty((self.N_inputs[k], self.num_comps),
                            dtype = self.access_buffers[k].dtype) \
            for k in self.accesses}

        self.retrieve_buffer_funcs = {}
        for k in self.accesses:
            self.retrieve_buffer_funcs[k] = \
                self.get_retrieve_buffer_func(
                    k, dtype = self.access_buffers[k].dtype) \
                if self.N_inputs[k] == 1 else \
                self.get_retrieve_buffer_multi_func(
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
            if self.N_inputs[k] == 1:
                self.retrieve_buffer(k, st = st)
            else:
                self.retrieve_buffer_multi(k, st = st)

        self.update_func.prepared_async_call(
            self.update_func.grid, self.update_func.block, st,
            self.num_comps, self.ddt*1000, self.nsteps,
            *[self.inputs[k].gpudata for k in self.accesses]+\
            [self.params_dict[k].gpudata for k in self.params]+\
            [self.internal_states[k].gpudata for k in self.internals]+\
            [update_pointers[k] for k in self.updates])

    def get_update_template(self):
        template = """
#include "stdio.h"
__global__ void exponential_with_stdp(int num_comps, %(dt)s dt, int steps,
                       %(spike_state)s* g_spike_state,
                       %(tau_g)s* g_tau_g,
                       %(tau_plus)s* g_tau_plus,
                       %(tau_minus)s* g_tau_minus,
                       %(eta_plus)s* g_eta_plus,
                       %(eta_minus)s* g_eta_minus,
                       %(w_max)s* g_w_max,
                       %(w_min)s* g_w_min,
                       %(internal_g)s* g_internal_g,
                       %(weight)s* g_weight,
                       %(x)s* g_x,
                       %(y)s* g_y,
                       %(g)s* g_g)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    %(spike_state)s spike_pre, spike_post;
    %(tau_g)s tau_g;
    %(tau_plus)s tau_plus;
    %(tau_minus)s tau_minus;
    %(eta_plus)s eta_plus;
    %(eta_minus)s eta_minus;
    %(w_max)s w_max;
    %(w_min)s w_min;
    %(internal_g)s g;
    %(weight)s weight, dw;
    %(x)s x;
    %(y)s y;

    for(int i = tid; i < num_comps; i += total_threads)
    {
        spike_pre = g_spike_state[i];
        spike_post = g_spike_state[num_comps*1+i];

        tau_plus = g_tau_plus[i];
        tau_minus = g_tau_minus[i];
        eta_plus = g_eta_plus[i];
        eta_minus = g_eta_minus[i];
        w_max = g_w_max[i];
        w_min = g_w_min[i];
        weight = g_weight[i];
        x = g_x[i];
        y = g_y[i];

        dw = 0;
        if(spike_post)
        {
            dw += ((w_max - weight)*eta_plus)*x;
        }
        if(spike_pre)
        {
            dw -= ((weight-w_min)*eta_minus)*y;
        }
        weight += dt*dw;
        weight = fmax(fmin(weight, w_max), w_min);

        x = x*exp%(fletter)s(-dt/tau_plus);
        y = y*exp%(fletter)s(-dt/tau_minus);
        if(spike_pre)
        {
            x += 1;
        }
        if(spike_post)
        {
            y += 1;
        }

        tau_g = g_tau_g[i];
        g = g_internal_g[i]*exp%(fletter)s(-dt/tau_g);
        if(spike_pre)
        {
            g += weight/tau_g;
        }
        g_g[i] = g;
        g_internal_g[i] = g;
        g_weight[i] = weight;
        g_x[i] = x;
        g_y[i] = y;
    }
}
"""
        return template

    def get_update_func(self, dtypes):
        type_dict = {k: dtype_to_ctype(dtypes[k]) for k in dtypes}
        type_dict.update({'fletter': 'f' if type_dict['tau_g'] == 'tau_g' else ''})
        mod = SourceModule(self.get_update_template() % type_dict,
                           options=self.compile_options)
        func = mod.get_function("exponential_with_stdp")
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

    G.add_node('pre_neuron', **{
               'class': 'LeakyIAF',
               'name': 'pre_neuron',
               'resting_potential': -70.0,
               'threshold': -45.0,
               'reset_potential': -70.0,
               'capacitance': 0.07, # in mS
               'resistance': 1000.0, # in Ohm
               'initV': -68.0 # drive the two neurons with a slight spike time offset
               })

    G.add_node('post_neuron', **{
               'class': 'LeakyIAF',
               'name': 'post_neuron',
               'resting_potential': -70.0,
               'threshold': -45.0,
               'reset_potential': -70.0,
               'capacitance': 0.07, # in mS
               'resistance': 1000.0, # in Ohm
               'initV': -70.0 # drive the two neurons with a slight spike time offset
               })

    G.add_node('synapse0', **{
               'class': 'exponential_with_stdp',
               'name': 'exponential_with_stdp',
               'tau_g': 4.0,
               'tau_plus': 5.0,
               'tau_minus': 5.0,
               'eta_plus': 3.0,
               'eta_minus': 3.0,
               'w_max': 1.0,
               'w_min': 0.0,
               'reverse': 0.0
               })

    # order indicates if the spike input is from presynaptic or postsynaptic neuron
    G.add_edge('pre_neuron', 'synapse0', order = 0) # order will be used as double for now
    # For testing, do not connect the synapse to post neuron
    # G.add_edge('synapse0', 'post_neuron')
    G.add_edge('post_neuron', 'synapse0', order = 1)

    comp_dict, conns = LPU.graph_to_dicts(G)

    fl_input_processor = StepInputProcessor('I', ['pre_neuron', 'post_neuron'], 0.1, 0.0, 1.0)
    fl_output_processor = FileOutputProcessor(
                            [('g', None),
                             ('V', None), ('spike_state', None)],
                            'new_output.h5', sample_interval=1)

    man.add(LPU, 'ge', dt, comp_dict, conns,
            device=args.gpu_dev, input_processors = [fl_input_processor],
            output_processors = [fl_output_processor], debug=args.debug)

    man.spawn()
    man.start(steps=args.steps)
    man.wait()
