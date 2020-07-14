from neurokernel.LPU.NDComponents.SynapseModels.BaseSynapseModel import *

#This class assumes a single pre synaptic connection per component instance
class PowerGPotGPot(BaseSynapseModel):
    accesses = ['V']
    updates = ['g']
    params = ['threshold', 'slope', 'power', 'saturation']
    extra_params = []
    internals = OrderedDict([])

    @property
    def maximum_dt_allowed(self):
        return self.dt

    def get_update_template(self):
        template = """
__global__ void update(int num_comps, %(dt)s dt, int steps,
                       %(input_V)s* g_V, %(param_threshold)s* g_threshold,
                       %(param_slope)s* g_slope, %(param_power)s* g_power,
                       %(param_saturation)s* g_saturation,
                       %(update_g)s* g_g)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    %(dt)s ddt = dt*1000.; // s to ms
    %(input_V)s V;
    %(param_threshold)s threshold;
    %(param_slope)s slope;
    %(param_power)s power;
    %(param_saturation)s saturation;

    for(int i = tid; i < num_comps; i += total_threads)
    {
        V = g_V[i];
        threshold = g_threshold[i];
        slope = g_slope[i];
        power = g_power[i];
        saturation = g_saturation[i];

        g_g[i] = fmin%(fletter)s(saturation,
                    slope*pow%(fletter)s(fmax(0.0,V-threshold),power));
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

    from neurokernel.LPU.InputProcessors.RampInputProcessor import RampInputProcessor
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

    G.add_node('synapse0', **{
               'class': 'PowerGPotGPot',
               'name': 'PowerGPotGPot',
               'gmax': 0.4,
               'threshold': -55.0,
               'slope': 0.02,
               'power': 1.0,
               'saturation': 0.4,
               'reverse': 0.0
               })

    comp_dict, conns = LPU.graph_to_dicts(G)

    fl_input_processor = RampInputProcessor('V', ['synapse0'], 0.0, 1.0, -70.0, -30.0)
    fl_output_processor = FileOutputProcessor([('g', None)], 'new_output.h5', sample_interval=1)

    man.add(LPU, 'ge', dt, comp_dict, conns,
            device=args.gpu_dev, input_processors = [fl_input_processor],
            output_processors = [fl_output_processor], debug=args.debug)

    man.spawn()
    man.start(steps=args.steps)
    man.wait()
