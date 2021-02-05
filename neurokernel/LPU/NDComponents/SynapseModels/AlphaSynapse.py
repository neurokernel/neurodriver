from neurokernel.LPU.NDComponents.SynapseModels.BaseSynapseModel import *

class AlphaSynapse(BaseSynapseModel):
    accesses = ['spike_state'] # (bool)
    updates = ['g'] # conductance (mS/cm^2)
    params = ['gmax', # maximum conductance (mS/cm^2)
              'ar', # rise rate of conductance (ms^{-1})
              'ad', # decay rate of conductance (ms^{-1})
              ]
    internals = OrderedDict([('z', 0.0),  # g,
                             ('dz', 0.0),  # derivative of g
                             ('d2z', 0.0)  # second derivative of g
                             ])

    @property
    def maximum_dt_allowed(self):
        return 1e-4

    def get_update_template(self):
        # The following kernel assumes a maximum of one input connection
        # per neuron
        if self.internal_steps == 1:
            # this is a kernel that runs 1 step internally for each self.dt
            template = """
__global__ void update(int num_comps, %(dt)s dt, int nsteps,
                       %(input_spike_state)s* g_spike_state,
                       %(param_gmax)s* g_gmax, %(param_ar)s* g_ar,
                       %(param_ad)s* g_ad,
                       %(internal_z)s* g_z, %(internal_dz)s* g_dz,
                       %(internal_d2z)s* g_d2z, %(update_g)s* g_g)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    %(dt)s ddt = dt*1000.; // s to ms
    %(input_spike_state)s spike_state;
    %(param_gmax)s gmax;
    %(param_ar)s ar;
    %(param_ad)s ad;
    %(internal_z)s z, new_z;
    %(internal_dz)s dz, new_dz;
    %(internal_d2z)s d2z, new_d2z;

    for(int i = tid; i < num_comps; i += total_threads)
    {
        ar = g_ar[i];
        ad = g_ad[i];
        z = g_z[i];
        dz = g_dz[i];
        d2z = g_d2z[i];
        spike_state = g_spike_state[i];

        new_z = fmax( 0., z + ddt*dz );
        new_dz = dz + ddt*d2z;
        if( spike_state>0.0 )
            new_dz += ar*ad;
        new_d2z = -( ar+ad )*dz - ar*ad*z;

        gmax = g_gmax[i];
        g_z[i] = new_z;
        g_dz[i] = new_dz;
        g_d2z[i] = new_d2z;
        g_g[i] = new_z*gmax;
    }
}
"""
        else:
            # this is a kernel that runs self.nstep steps internally for each self.dt
            # see the "k" for loop
            template = """
__global__ void update(int num_comps, %(dt)s dt, int nsteps,
                       %(input_spike_state)s* g_spike_state,
                       %(param_gmax)s* g_gmax, %(param_ar)s* g_ar,
                       %(param_ad)s* g_ad,
                       %(internal_z)s* g_z, %(internal_dz)s* g_dz,
                       %(internal_d2z)s* g_d2z, %(update_g)s* g_g)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    %(dt)s ddt = dt*1000.; // s to ms
    %(input_spike_state)s spike_state;
    %(param_gmax)s gmax;
    %(param_ar)s ar;
    %(param_ad)s ad;
    %(internal_z)s z, new_z;
    %(internal_dz)s dz, new_dz;
    %(internal_d2z)s d2z, new_d2z;

    for(int i = tid; i < num_comps; i += total_threads)
    {
        ar = g_ar[i];
        ad = g_ad[i];
        z = g_z[i];
        dz = g_dz[i];
        d2z = g_d2z[i];
        spike_state = g_spike_state[i];

        for(int k = 0; k < nsteps; ++k)
        {
            new_z = fmax( 0., z + ddt*dz );
            new_dz = dz + ddt*d2z;
            if(k == 0 && (spike_state>0.0))
                new_dz += ar*ad;
            new_d2z = -( ar+ad )*dz - ar*ad*z;

            z = new_z;
            dz = new_dz;
            d2z = new_d2z;
        }

        gmax = g_gmax[i];
        g_z[i] = new_z;
        g_dz[i] = new_dz;
        g_d2z[i] = new_d2z;
        g_g[i] = new_z*gmax;
    }
}
"""
        return template

if __name__ == '__main__':
    import argparse
    import itertools

    import networkx as nx
    import h5py

    from neurokernel.tools.logging import setup_logger
    import neurokernel.core_gpu as core
    from neurokernel.LPU.LPU import LPU
    from neurokernel.LPU.InputProcessors.FileInputProcessor import FileInputProcessor
    from neurokernel.LPU.OutputProcessors.FileOutputProcessor import FileOutputProcessor
    import neurokernel.mpi_relaunch

    dt = 1e-4
    dur = 1.0
    steps = int(dur / dt)

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=False,
                        dest='debug', action='store_true',
                        help='Write connectivity structures and inter-LPU routed data in debug folder')
    parser.add_argument('-l', '--log', default='both', type=str,
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

    t = np.arange(0, dt * steps, dt)

    uids = np.array(["synapse0"], dtype='S')

    spike_state = np.zeros((steps, 1), dtype=np.float64)
    spike_state[np.nonzero((t - np.round(t / 0.04) * 0.04) == 0)[0]] = 1

    with h5py.File('input_spike.h5', 'w') as f:
        f.create_dataset('spike_state/uids', data=uids)
        f.create_dataset('spike_state/data', (steps, 1),
                         dtype=np.float64,
                         data=spike_state)

    man = core.Manager()

    G = nx.MultiDiGraph()

    G.add_node('synapse0', **{
               'class': 'AlphaSynapse',
               'name': 'AlphaSynapse',
               'gmax': 0.003,
               'ar': 0.11,
               'ad': 0.19,
               'reverse': 0.0
               })

    comp_dict, conns = LPU.graph_to_dicts(G)

    fl_input_processor = FileInputProcessor('input_spike.h5')
    fl_output_processor = FileOutputProcessor(
        [('g', None)], 'new_output.h5', sample_interval=1)

    man.add(LPU, 'ge', dt, comp_dict, conns,
            device=args.gpu_dev, input_processors=[fl_input_processor],
            output_processors=[fl_output_processor], debug=args.debug)

    man.spawn()
    man.start(steps=args.steps)
    man.wait()
