
import time
import argparse
import itertools
import networkx as nx
from neurokernel.tools.logging import setup_logger
import neurokernel.core_gpu as core

from neurokernel.LPU.LPU import LPU

from neurokernel.LPU.InputProcessors.FileInputProcessor import FileInputProcessor
from neurokernel.LPU.InputProcessors.StepInputProcessor import StepInputProcessor
from neurokernel.LPU.OutputProcessors.FileOutputProcessor import FileOutputProcessor
from neurokernel.LPU.OutputProcessors.OutputRecorder import OutputRecorder

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

N = 1024
for i in range(N):
    G.add_node('neuron{}'.format(i), **{
               'class': 'HodgkinHuxley2',
               'name': 'HodgkinHuxley2',
               'g_K': 36.0,
               'g_Na': 120.0,
               'g_L': 0.3,
               'E_K': -77.0,
               'E_Na': 50.0,
               'E_L': -54.387
               })

comp_dict, conns = LPU.graph_to_dicts(G)


fl_input_processor = StepInputProcessor('I', ['neuron{}'.format(i) for i in range(N)], 20.0, 0.0, dur)
fl_output_processor = [FileOutputProcessor([('spike_state', None), ('V', None)], 'output.h5', sample_interval=1)]

#fl_output_processor = [OutputRecorder([('spike_state', None), ('V', None)], dur, dt, sample_interval = 1)]

man.add(LPU, 'ge', dt, comp_dict, conns,
        device=args.gpu_dev, input_processors=[fl_input_processor],
        output_processors=fl_output_processor, debug=args.debug,
        print_timing = True, time_sync = False,
        extra_comps=[HodgkinHuxley2])

man.spawn()
man.start(steps=args.steps)
start_time = time.time()
man.wait()
end_time = time.time()
print(end_time-start_time)
