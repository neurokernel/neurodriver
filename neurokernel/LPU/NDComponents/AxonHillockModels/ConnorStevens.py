
from collections import OrderedDict

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from BaseAxonHillockModel import BaseAxonHillockModel

class ConnorStevens(BaseAxonHillockModel):
    updates = ['spike_state', 'V']
    accesses = ['I']
    params = ['n','m','h','a','b']
    internals = OrderedDict([('internalV',-65.)])

    def __init__(self, params_dict, access_buffers, dt,
                 debug=False, LPU_id=None, cuda_verbose=True):
        if cuda_verbose:
            self.compile_options = ['--ptxas-options=-v']
        else:
            self.compile_options = []

        self.num_comps = params_dict['n'].size
        self.params_dict = params_dict
        self.access_buffers = access_buffers

        self.debug = debug
        self.LPU_id = LPU_id
        self.dtype = params_dict['n'].dtype

        self.dt = np.double(dt)
        self.ddt = np.double(1e-6)
        self.steps = np.int32(max( int(self.dt/self.ddt), 1 ))

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
        dtypes.update({k: self.dtype if not k == 'spike_state' else np.int32 for k in self.updates})
        self.update_func = self.get_update_func(dtypes)

    def pre_run(self, update_pointers):
        if self.params_dict.has_key('initV'):
            cuda.memcpy_dtod(int(update_pointers['V']),
                             self.params_dict['initV'].gpudata,
                             self.params_dict['initV'].nbytes)
            cuda.memcpy_dtod(self.internal_states['internalV'].gpudata,
                             self.params_dict['initV'].gpudata,
                             self.params_dict['initV'].nbytes)


    def run_step(self, update_pointers, st=None):
        for k in self.inputs:
            self.sum_in_variable(k, self.inputs[k], st=st)

        self.update_func.prepared_async_call(
            self.update_func.grid, self.update_func.block, st,
            self.num_comps, self.ddt, self.steps,
            *[self.inputs[k].gpudata for k in self.accesses]+\
            [self.params_dict[k].gpudata for k in self.params]+\
            [self.internal_states[k].gpudata for k in self.internals]+\
            [update_pointers[k] for k in self.updates])

    def get_update_template(self):
        template = """
#define EXP exp%(fletter)s
#define POW pow%(fletter)s

#define		E_K		-72.
#define		E_Na		55.
#define		E_a		-75.
#define		E_l		-17.
#define		G_total		67.7
#define		G_a		47.7
#define		G_Na		120.
#define		G_K		(G_total-G_a)
#define		G_l		0.3
#define		ms		-5.3
#define		hs		-12.
#define		ns		-4.3

__global__ void update(
    int num_comps,
    %(dt)s dt,
    int nsteps,
    %(I)s* g_I,
    %(n)s* g_n,
    %(m)s* g_m,
    %(h)s* g_h,
    %(a)s* g_a,
    %(b)s* g_b,
    %(internalV)s* g_internalV,
    %(spike_state)s* g_spike_state,
    %(V)s* g_V)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    %(dt)s ddt = dt*1000.; // s to ms

    %(V)s V, Vprev1, Vprev2;
    %(I)s I;
    %(spike_state)s spike;

    %(n)s n, a_n, b_n, n_inf, tau_n;
    %(m)s m, a_m, b_m, m_inf, tau_m;
    %(h)s h, a_h, b_h, h_inf, tau_h;
    %(a)s a, a_inf, tau_a;
    %(b)s b, b_inf, tau_b;


    for(int i = tid; i < num_comps; i += total_threads)
    {
        spike = 0;
        V = g_internalV[i];
        I = g_I[i];
        n = g_n[i];
        m = g_m[i];
        h = g_h[i];
        a = g_a[i];
        b = g_b[i];

        for (int j = 0; j < nsteps; ++j)
        {
            /*
             * Hodgkin-Huxley with shifts - 3.8 is temperature factor
             */
            a_m = -.1*(V+35+ms)/(EXP(-(V+35+ms)/10)-1);
            b_m = 4*EXP(-(V+60+ms)/18);
            m_inf = a_m/(a_m+b_m);
            tau_m = 1/(3.8*(a_m+b_m));

            a_h = .07*EXP(-(V+60+hs)/20);
            b_h = 1/(1+EXP(-(V+30+hs)/10));
            h_inf = a_h/(a_h+b_h);
            tau_h = 1/(3.8*(a_h+b_h));

            a_n = -.01*(V+50+ns)/(EXP(-(V+50+ns)/10)-1);
            b_n = .125*EXP(-(V+60+ns)/80);
            n_inf = a_n/(a_n+b_n);
            tau_n = 2/(3.8*(a_n+b_n));

            a_inf = POW(.0761*EXP((V+94.22)/31.84)/(1+EXP((V+1.17)/28.93)),.3333);
            tau_a = .3632+1.158/(1+EXP((V+55.96)/20.12));
            b_inf = POW(1/(1+EXP((V+53.3)/14.54)),4);
            tau_b = 1.24+2.678/(1+EXP((V+50)/16.027));

            V += ddt*(I-G_l*(V-E_l)-G_Na*h*m*m*m*(V-E_Na)-G_K*n*n*n*n*(V-E_K)-G_a*b*a*a*a*(V-E_a));
            m += ddt*(m_inf-m)/tau_m;
            h += ddt*(h_inf-h)/tau_h;
            n += ddt*(n_inf-n)/tau_n;
            a += ddt*(a_inf-a)/tau_a;
            b += ddt*(b_inf-b)/tau_b;

            spike += (Vprev2<=Vprev1) && (Vprev1 >= V) && (Vprev1 > -30);

            Vprev2 = Vprev1;
            Vprev1 = V;

        }
        g_n[i] = n;
        g_m[i] = m;
        g_h[i] = h;
        g_a[i] = a;
        g_b[i] = b;
        g_V[i] = V;
        g_internalV[i] = V;
        g_spike_state[i] = (spike > 0);
    }
}
"""
        return template

    def get_update_func(self, dtypes):
        type_dict = {k: dtype_to_ctype(dtypes[k]) for k in dtypes}
        type_dict.update({'fletter': 'f' if type_dict['n'] == 'float' else ''})
        mod = SourceModule(self.get_update_template() % type_dict,
                           options=self.compile_options)
        func = mod.get_function("update")
        func.prepare('i'+np.dtype(dtypes['dt']).char+'i'+'P'*(len(type_dict)-2))
        func.block = (128,1,1)
        func.grid = (min(6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT,
                         (self.num_comps-1) / 128 + 1), 1)
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
               'class': 'ConnorStevens',
               'name': 'ConnorStevens',
               'n': 0.,
               'm': 0.,
               'h': 1.,
               'a': 0.,
               'b': 0.,
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

    # plot the result
    import h5py
    import matplotlib
    matplotlib.use('PS')
    import matplotlib.pyplot as plt

    f = h5py.File('new_output.h5')
    t = np.arange(0, args.steps)*dt

    plt.figure()
    plt.plot(t,f['V'].values()[0])
    plt.xlabel('time, [s]')
    plt.ylabel('Voltage, [mV]')
    plt.title('Connor-Stevens Neuron')
    plt.savefig('csn.png',dpi=300)
