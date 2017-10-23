from BaseSynapseModel import BaseSynapseModel
from collections import OrderedDict
import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# The following kernel assumes a maximum of one input connection
# per neuron

class AlphaSynapse(BaseSynapseModel):
    accesses = ['spike_state']
    states = OrderedDict([('a0', 0.), ('a1', 0.), ('a2', 0.)])
    params = OrderedDict([('ar', 1.), ('ad', 1.), ('gmax', 1.0),
        ('reverse', -65.)])
    cuda_src = cuda_src = cuda_src = """
# if (defined(USE_DOUBLE))
#    define FLOATTYPE double
#    define EXP exp
#    define POW pow
#    define FMAX fmax
# else
#    define FLOATTYPE float
#    define EXP expf
#    define POW powf
#    define FMAX fmaxf
# endif
#
# if (defined(USE_LONG_LONG))
#     define INTTYPE long long
# else
#     define INTTYPE int
# endif
#

__global__ void alpha_synapse(
    int num,
    FLOATTYPE dt,
    INTTYPE *spike,
    INTTYPE ld,
    INTTYPE current,
    INTTYPE buffer_length,
    FLOATTYPE *Ar,
    FLOATTYPE *Ad,
    FLOATTYPE *Gmax,
    FLOATTYPE *a0,
    FLOATTYPE *a1,
    FLOATTYPE *a2,
    FLOATTYPE *cond,
    INTTYPE *Pre,
    INTTYPE *npre,
    INTTYPE *cumpre,
    INTTYPE *delay)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int tot_threads = gridDim.x * blockDim.x;
    FLOATTYPE ar,ad,gmax;
    FLOATTYPE old_a[3];
    FLOATTYPE new_a[3];
    INTTYPE pre;

    INTTYPE col;
    for (int i=tid; i<num; i+=tot_threads) {
        // copy data from global memory to register
        if(npre[i]){
            ar = Ar[i];
            ad = Ad[i];
            gmax = Gmax[i];
            old_a[0] = a0[i];
            old_a[1] = a1[i];
            old_a[2] = a2[i];
            // update the alpha function
            new_a[0] = FMAX( 0., old_a[0] + dt*old_a[1] );
            new_a[1] = old_a[1] + dt*old_a[2];
            col = current-delay[i];
            if (col < 0)
                col = buffer_length + col;
            pre = col*ld + Pre[cumpre[i]];
            if (spike[pre])
                new_a[1] += ar*ad;
            new_a[2] = -( ar+ad )*old_a[1] - ar*ad*old_a[0];


            // copy data from register to the global memory
            a0[i] = new_a[0];
            a1[i] = new_a[1];
            a2[i] = new_a[2];
            cond[i] = new_a[0]*gmax;
        } else
            cond[i] = 0;
    }
    return;
}
"""

    def run_step(self, update_pointers, st = None):
        self.update_func.prepared_async_call(
            self.update_func.gpu_grid,
            self.update_func.gpu_block,
            st,
            self.num_comps,
            self.dt,
            self.access_buffers['spike_state'].gpudata,
            self.access_buffers['spike_state'].ld,
            self.access_buffers['spike_state'].current,
            self.access_buffers['spike_state'].buffer_length,
            self.params_dict['ar'].gpudata,
            self.params_dict['ad'].gpudata,
            self.params_dict['gmax'].gpudata,
            self.states['a0'].gpudata,
            self.states['a1'].gpudata,
            self.states['a2'].gpudata,
            update_pointers['g'],
            self.params_dict['pre']['spike_state'].gpudata,
            self.params_dict['npre']['spike_state'].gpudata,
            self.params_dict['cumpre']['spike_state'].gpudata,
            self.params_dict['conn_data']['spike_state']['delay'].gpudata)

    def get_update_func(self):
        mod = SourceModule(self.cuda_src, options=self.compile_options)
        func = mod.get_function("alpha_synapse")
        func.prepare(
            np.dtype(self.inttype).char+np.dtype(self.floattype).char+'P' + \
            np.dtype(self.inttype).char*3 + 'P'*11)

        func.gpu_block = (128,1,1)
        func.gpu_grid = (min( 6*cuda.Context.get_device().MULTIPROCESSOR_COUNT,\
                              (self.num_comps-1)/128 + 1), 1)
        return func

if __name__ == '__main__':
    import argparse
    import itertools
    import h5py
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

    G.add_node('Synapse0', **{
               'class': 'AlphaSynapse',
               'name': 'AlphaSynapse',
               'ar': 100.0,
               'ad': 100.0,
               'gmax': .003,
               'reverse': -65.,
               })

    comp_dict, conns = LPU.graph_to_dicts(G)

    spike_train  = np.zeros(steps, dtype=np.int32)
    spike_train[::100] = 1


    with h5py.File('input.h5', 'w') as f:
        f.create_dataset('spike_state/uids', data=['Synapse0'])
        f.create_dataset('spike_state/data', (args.steps, 1),
                         dtype=np.int32, data=spike_train)

    fl_input_processor = FileInputProcessor('input.h5')
    fl_output_processor = FileOutputProcessor([('g', None)], 'new_output.h5', sample_interval=1)

    man.add(LPU, 'ge', dt, comp_dict, conns, cuda_verbose=True,
            device=args.gpu_dev, input_processors = [fl_input_processor],
            output_processors = [fl_output_processor], debug=args.debug)

    man.spawn()
    man.start(steps=args.steps)
    man.wait()

    # plot the result
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    f = h5py.File('new_output.h5')
    t = np.arange(0, args.steps)*dt

    plt.figure()
    plt.subplot(211)
    spk = spike_train.flatten().nonzero()[0]
    plt.stem(t[spk],np.ones((len(spk),)))
    plt.title('Spike Train Stimulus')
    plt.xlabel('time, [s]')
    plt.ylabel('Spike')
    plt.xlim([0, dur])
    plt.ylim([0, 1.2])
    plt.grid()

    plt.subplot(212)
    plt.plot(t,f['g'].values()[0])
    plt.xlabel('time, [s]')
    plt.ylabel('Conductance, [mS]')
    plt.title('AlphaSynapse')
    plt.xlim([0, dur])
    plt.grid()
    plt.tight_layout()
    plt.savefig('alpha_synapse.png',dpi=300)
