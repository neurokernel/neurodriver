from neurokernel.LPU.NDComponents.AxonHillockModels.BaseAxonHillockModel import *

class LeakyIAF(BaseAxonHillockModel):
    # updates are the output states of the model
    updates = ['spike_state', # (bool)
               'V' # Membrane Potential (mV)
              ]
    # accesses are the input variables of the model
    accesses = ['I'] # (\mu A/cm^2 )
    # params are the parameters of the model that needs to be defined
    # during specification of the model
    params = ['resting_potential', # (mV)
              'threshold', # Firing Threshold (mV)
              'reset_potential', # Potential to be reset to after a spike (mV)
              'capacitance', # (\mu F/cm^2)
              'resistance' # (k\Omega cm.^2)
              ]
    extra_params = ['initV']
    # internals are the variables used to store internal states of the model,
    # and are ordered dict whose keys are the variables and value are the initial values.
    internals = OrderedDict([('V', 0.0)]) # Membrane Potential (mV)

    @property
    def maximum_dt_allowed(self):
        return 1e-4

    def pre_run(self, update_pointers):
        super(LeakyIAF, self).pre_run(update_pointers)
        if 'initV' not in self.params_dict:
            self.add_initializer('resting_potential', 'V', update_pointers)

    def get_update_template(self):
        # need to update the CUDA kernel to reflect the equations of the model
        # the argument of the function must in the following order:
        # 1. int num_comps
        # 2. %(dt)s dt,
        # 3. int nsteps,
        # 4. all variables in accesses according to the order in accesses
        # 5. all variables in params according to the order in params
        # 6. all variables in internals according to the order in internals
        # 7. all variables in updates according to the order in updates
        template = """
__global__ void update(int num_comps,
               %(dt)s dt,
               int nsteps,
               %(input_I)s* g_I, // accesses
               %(param_resting_potential)s* g_resting_potential, // params
               %(param_threshold)s* g_threshold,
               %(param_reset_potential)s* g_reset_potential,
               %(param_capacitance)s* g_capacitance,
               %(param_resistance)s* g_resistance,
               %(internal_V)s* g_internalV, // internals
               %(update_spike_state)s* g_spike_state, //updates
               %(update_V)s* g_V)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    // instantiate variables
    %(dt)s ddt = dt*1000.; // s to ms
    %(update_V)s V;
    %(input_I)s I;
    %(update_spike_state)s spike;
    %(param_resting_potential)s resting_potential;
    %(param_threshold)s threshold;
    %(param_reset_potential)s reset_potential;
    %(param_capacitance)s capacitance;
    %(param_resistance)s resistance;
    %(dt)s bh;

    // no need to change this for loop
    for(int i = tid; i < num_comps; i += total_threads)
    {
        // load the data from global memory
        spike = 0;
        V = g_internalV[i];
        I = g_I[i];
        capacitance = g_capacitance[i];
        resting_potential = g_resting_potential[i];
        threshold = g_threshold[i];
        resistance = g_resistance[i];
        reset_potential = g_reset_potential[i];

        // update according to equations of the model
        bh = exp%(fletter)s(-ddt/(capacitance*resistance));

        for (int j = 0; j < nsteps; ++j)
        {
            V = V*bh + (resistance*I+resting_potential)*(1.0 - bh);
            if (V >= threshold)
            {
                V = reset_potential;
                spike = 1.0;
            }
        }

        // write local updated states back to global memory
        g_V[i] = V;
        g_internalV[i] = V;
        g_spike_state[i] = spike;
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

    # specify the node
    G.add_node('neuron0', **{
               'class': 'LeakyIAF',
               'name': 'LeakyIAF',
               'resting_potential': -70.0,
               'reset_potential': -70.0,
               'threshold': -45.0, #
               'capacitance': 2.0, # in mS
               'resistance': 10.0, # in Ohm
               })

    comp_dict, conns = LPU.graph_to_dicts(G)

    # use a input processor that present a step current (I) input to 'neuron0'
    # the step is from 0.2 to 0.8 and the step height is 10.0
    fl_input_processor = StepInputProcessor('I', ['neuron0'], 10.0, 0.2, 0.8)
    # output processor to record 'spike_state' and 'V' to hdf5 file 'new_output.h5',
    # with a sampling interval of 1 run step.
    fl_output_processor = FileOutputProcessor([('spike_state', None),('V', None)], 'new_output.h5', sample_interval=1)

    man.add(LPU, 'ge', dt, comp_dict, conns,
            device=args.gpu_dev, input_processors = [fl_input_processor],
            output_processors = [fl_output_processor], debug=args.debug)

    man.spawn()
    man.start(steps=args.steps)
    man.wait()

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
    plt.title('Leaky Integrate-and-Fire Neuron')
    plt.savefig('lif.png',dpi=300)
