from neurokernel.LPU.NDComponents.AxonHillockModels.BaseAxonHillockModel import *

class HodgkinHuxley(BaseAxonHillockModel):
    updates = ['spike_state', # (bool)
               'V' # Membrane Potential (mV)
              ]
    accesses = ['I'] # Current (\mu A/cm^2)
    params = ['n', # state variable for activation of K channel ([0-1] unitless)
              'm', # state variable for activation of Na channel ([0-1] unitless)
              'h'  # state variable for inactivation of Na channel ([0-1] unitless)
              ]
    extra_params = ['initV']
    internals = OrderedDict([('V',-65.),  # Membrane Potential (mV)
                             ('Vprev1',-65.)]) # Membrane Potential (mV)

    @property
    def maximum_dt_allowed(self):
        return 1e-5

    def pre_run(self, update_pointers):
        super(HodgkinHuxley, self).pre_run(update_pointers)
        # if 'initV' in self.params_dict:
        self.add_initializer('initV', 'Vprev1', update_pointers)

    def get_update_template(self):
        template = """
#define EXP exp%(fletter)s
#define POW pow%(fletter)s
#define ABS fabs%(fletter)s
#define MIN fmin%(fletter)s

__global__ void update(
    int num_comps,
    %(dt)s dt,
    int nsteps,
    %(input_I)s* g_I,
    %(param_n)s* g_n,
    %(param_m)s* g_m,
    %(param_h)s* g_h,
    %(internal_V)s* g_internalV,
    %(internal_Vprev1)s* g_Vprev1,
    %(update_spike_state)s* g_spike_state,
    %(update_V)s* g_V)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    %(dt)s ddt = dt*1000.; // s to ms

    %(update_V)s V, Vprev1, Vprev2, dV;
    %(input_I)s I;
    %(update_spike_state)s spike;
    %(param_n)s n, dn;
    %(param_m)s m, dm;
    %(param_h)s h, dh;
    %(param_n)s a;

    for(int i = tid; i < num_comps; i += total_threads)
    {
        spike = 0.0;
        V = g_internalV[i];
        Vprev1 = V;
        Vprev2 = g_Vprev1[i];
        I = g_I[i];
        n = g_n[i];
        m = g_m[i];
        h = g_h[i];

        for (int j = 0; j < nsteps; ++j)
        {
            a = exp(-(V+55)/10)-1;
            if (ABS(a) <= 1e-7)
                dn = (1.-n) * 0.1 - n * (0.125*EXP(-(V+65.)/80.));
            else
                dn = (1.-n) * (-0.01*(V+55.)/a) - n * (0.125*EXP(-(V+65)/80));

            a = exp(-(V+40.)/10.)-1.;
            if (ABS(a) <= 1e-7)
                dm = (1.-m) - m*(4*EXP(-(V+65)/18));
            else
                dm = (1.-m) * (-0.1*(V+40.)/a) - m * (4.*EXP(-(V+65.)/18.));

            dh = (1.-h) * (0.07*EXP(-(V+65.)/20.)) - h / (EXP(-(V+35.)/10.)+1.);

            dV = I - 120.*POW(m,3)*h*(V-50.) - 36. * POW(n,4) * (V+77.) - 0.3 * (V+54.387);

            n += ddt * dn;
            m += ddt * dm;
            h += ddt * dh;
            V += ddt * dV;

            spike += (Vprev2<=Vprev1) && (Vprev1 >= V) && (Vprev1 > -30);

            Vprev2 = Vprev1;
            Vprev1 = V;
        }

        g_n[i] = n;
        g_m[i] = m;
        g_h[i] = h;
        g_V[i] = V;
        g_internalV[i] = Vprev1;
        g_Vprev1[i] = Vprev2;
        g_spike_state[i] = MIN(spike, 1.0);
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

    G.add_node('neuron0', **{
               'class': 'HodgkinHuxley',
               'name': 'HodgkinHuxley',
               'n': 0.,
               'm': 0.,
               'h': 1.,
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
    plt.plot(t,list(f['V'].values())[0])
    plt.xlabel('time, [s]')
    plt.ylabel('Voltage, [mV]')
    plt.title('Hodgkin-Huxley Neuron')
    plt.savefig('hhn.png',dpi=300)
