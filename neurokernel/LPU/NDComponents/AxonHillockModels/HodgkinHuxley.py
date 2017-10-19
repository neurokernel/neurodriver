
from collections import OrderedDict

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from BaseAxonHillockModel import BaseAxonHillockModel

class HodgkinHuxley(BaseAxonHillockModel):
    updates = ['spike_state', 'V']
    accesses = ['I']
    params = OrderedDict()
    states = OrderedDict([
        ('n', 0.),
        ('m', 0.),
        ('h', 1.),
        ('V',-65.)])
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

__global__ void update(
    int num_comps,
    FLOATTYPE dt,
    int nsteps,
    FLOATTYPE* g_I,
    FLOATTYPE* g_n,
    FLOATTYPE* g_m,
    FLOATTYPE* g_h,
    FLOATTYPE* g_internalV,
    INTTYPE* g_spike_state,
    FLOATTYPE* g_V)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    FLOATTYPE V, Vprev1, Vprev2, dV;
    FLOATTYPE I;
    INTTYPE spike;
    FLOATTYPE n, dn;
    FLOATTYPE m, dm;
    FLOATTYPE h, dh;
    FLOATTYPE a;

    for(int i = tid; i < num_comps; i += total_threads)
    {
        spike = 0;
        V = g_internalV[i];
        I = g_I[i];
        n = g_n[i];
        m = g_m[i];
        h = g_h[i];

        for (int j = 0; j < nsteps; ++j)
        {
            a = exp(-(V+55)/10)-1;
            if (abs(a) <= 1e-7)
                dn = (1.-n) * 0.1 - n * (0.125*EXP(-(V+65.)/80.));
            else
                dn = (1.-n) * (-0.01*(V+55.)/a) - n * (0.125*EXP(-(V+65)/80));

            a = exp(-(V+40.)/10.)-1.;
            if (abs(a) <= 1e-7)
                dm = (1.-m) - m*(4*EXP(-(V+65)/18));
            else
                dm = (1.-m) * (-0.1*(V+40.)/a) - m * (4.*EXP(-(V+65.)/18.));

            dh = (1.-h) * (0.07*EXP(-(V+65.)/20.)) - h / (EXP(-(V+35.)/10.)+1.);

            dV = I - 120.*POW(m,3)*h*(V-50.) - 36. * POW(n,4) * (V+77.) - 0.3 * (V+54.387);

            n += dt * dn;
            m += dt * dm;
            h += dt * dh;
            V += dt * dV;

            spike += (Vprev2<=Vprev1) && (Vprev1 >= V) && (Vprev1 > -30);

            Vprev2 = Vprev1;
            Vprev1 = V;
        }

        g_n[i] = n;
        g_m[i] = m;
        g_h[i] = h;
        g_V[i] = V;
        g_internalV[i] = V;
        g_spike_state[i] = (spike > 0);
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
               'class': 'HodgkinHuxley',
               'name': 'HodgkinHuxley'
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
    plt.title('Hodgkin-Huxley Neuron')
    plt.savefig('hhn.png',dpi=300)
