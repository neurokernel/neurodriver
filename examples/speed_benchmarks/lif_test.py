
import time
import argparse
import itertools
import networkx as nx
import pickle
import pycuda.driver as cuda
from neurokernel.tools.logging import setup_logger
import neurokernel.core_gpu as core

from neurokernel.LPU.LPU import LPU

from neurokernel.LPU.InputProcessors.FileInputProcessor import FileInputProcessor
from neurokernel.LPU.InputProcessors.StepInputProcessor import StepInputProcessor
from neurokernel.LPU.OutputProcessors.FileOutputProcessor import FileOutputProcessor
from neurokernel.LPU.OutputProcessors.OutputRecorder import OutputRecorder

import neurokernel.mpi_relaunch

dt = 1e-4
dur = 10.0
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

start_time = time.time()
G = nx.MultiDiGraph()

N = 1024
for i in range(N):
    G.add_node('neuron{}'.format(i), **{
               'class': 'LeakyIAF',
               'name': 'LeakyIAF',
               'resting_potential': -70.0, # (mV)
              'threshold': -40.0, # Firing Threshold (mV)
              'reset_potential': -70.0, # Potential to be reset to after a spike (mV)
              'capacitance': 1, # (\mu F/cm^2)
              'resistance': 0.007 # (k\Omega cm.^2)
               })
print("Creating graph completed in {} seconds.".format(time.time()-start_time))

fl_input_processor = StepInputProcessor('I', ['neuron{}'.format(i) for i in range(N)], 20.0, 0.0, dur)
# Write result to disk
#fl_output_processor = [FileOutputProcessor([('spike_state', None), ('V', None)], 'output.h5', sample_interval=1)]

# Result in memory
fl_output_processor = [OutputRecorder([('spike_state', None), ('V', None)], sample_interval = 1)]


man.add(LPU, 'ge', dt, 'pickle', pickle.dumps(G),
        device=args.gpu_dev, input_processors=[fl_input_processor],
        output_processors=fl_output_processor, debug=args.debug,
        time_sync = False, print_timing = False,
        extra_comps=[])
print("Adding LPU completed in {} seconds.".format(time.time()-start_time))

start_time = time.time()
man.spawn()
man.start(steps=args.steps)
print("Spawning LPUs Completed in {} seconds.".format(time.time()-start_time))
execution_time = man.timed_wait()
compile_and_execution_time = time.time()-start_time
print("LPU Execution Completed in {} seconds.".format(execution_time))
print("LPUs Compilation and Execution Completed in {} seconds.".format(compile_and_execution_time))

