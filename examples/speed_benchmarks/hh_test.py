
import time
import argparse
import itertools
import dill
import pickle
import functools
import networkx as nx
import numpy as np
from neurokernel.tools.logging import setup_logger
import neurokernel.core_gpu as core

from neurokernel.LPU.LPU import LPU

from neurokernel.LPU.InputProcessors.FileInputProcessor import FileInputProcessor
from neurokernel.LPU.InputProcessors.StepInputProcessor import StepInputProcessor
from neurokernel.LPU.OutputProcessors.FileOutputProcessor import FileOutputProcessor
from neurokernel.LPU.OutputProcessors.OutputRecorder import OutputRecorder

import neurokernel.mpi_relaunch


parser = argparse.ArgumentParser()
parser.add_argument('--debug', default=False,
                    dest='debug', action='store_true',
                    help='Write connectivity structures and inter-LPU routed data in debug folder')
parser.add_argument('-l', '--log', default='none', type=str,
                    help='Log output to screen [file, screen, both, or none; default:none]')
# parser.add_argument('-s', '--steps', default=steps, type=int,
#                     help='Number of steps [default: %s]' % steps)
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


def create_graph(N):
    G = nx.MultiDiGraph()

    # Create N HodgkinHuxley2 neurons
    for i in range(N):
        id = 'neuron_{}'.format(i)

        n = np.random.rand()
        G.add_node(id,
                **{'class': 'HodgkinHuxley2',
                    'name': 'HodgkinHuxley2',
                    'g_K': 36.0,
                    'g_Na': 120.0,
                    'g_L': 0.3,
                    'E_K': -77.0,
                    'E_Na': 50.0,
                    'E_L': -54.387,
                    'initV': np.random.rand()*70-70,
                    'initn': n,
                    'initm': np.random.rand(),
                    'inith': 0.89-1.1*n
                    })
    return G

def simulation(dt, N, output_n, nsteps = 10000):
    start_time = time.time()

    dur = nsteps * dt
    steps = nsteps

    man = core.Manager()

    G = create_graph(N)
    print("Creating graph completed in {} seconds.".format(time.time()-start_time))

    start_time = time.time()
    #comp_dict, conns = LPU.graph_to_dicts(G, remove_edge_id=False)

    fl_input_processor = StepInputProcessor('I', ['neuron_{}'.format(i) for i in range(N)], 20.0, 0.0, dur)
    fl_output_processor = [FileOutputProcessor([('V', None), ('spike_state', None)],
                                               'neurodriver_output_{}.h5'.format(output_n),
                                               sample_interval=1, cache_length=2000)]
    #fl_output_processor = [] # temporarily suppress generating output

    #fl_output_processor = [OutputRecorder([('spike_state', None), ('V', None), ('g', None), ('E', None)], dur, dt, sample_interval = 1)]

    man.add(LPU, 'ge', dt, 'pickle', pickle.dumps(G),
            device=args.gpu_dev, input_processors=[fl_input_processor],
            output_processors=fl_output_processor, debug=args.debug,
            print_timing=False, time_sync=False,
            extra_comps=[])
    print("Adding LPU completed in {} seconds.".format(time.time()-start_time))

    start_time = time.time()
    man.spawn()
    print('DEBUG: #simulation steps={}'.format(steps))
    man.start(steps=steps)
    print("Spawning LPUs Completed in {} seconds.".format(time.time()-start_time))
    start_time = time.time()
    execution_time = man.wait(return_timing = True)
    compile_and_execute_time = time.time()-start_time
    print("LPUs Compilation and Execution Completed in {} seconds.".format(compile_and_execute_time))
    return compile_and_execute_time, execution_time

if __name__ == '__main__':
    sim_time = []
    compile_and_sim_time = []

    # comparison 1:
    diff_dt = [1e-6] # run at the internal dt used by HH2.

    # comparison 2:
    #diff_dt = [1e-4] # Keep ddt at 1e-6 all the time.
                     # The ODEs must be run at small dt,
                     # but no need to store result for every 1e-6 second.
                     # Also couple with sample_interval in FileOutputProcessor
                     # to further reduce the storage.
    diff_N = [2, 32, 128, 256, 512]
    n_sim = 1
    i = 0
    for dt in diff_dt:
        for N in diff_N:
            sim_time.append([])
            compile_and_sim_time.append([])
            for t in range(n_sim + 1):
                c, s = simulation(dt, N, i * n_sim + t, 10000)
                sim_time[i].append(s)
                compile_and_sim_time[i].append(c)
            sim_time[i].pop(0) # discard first result
            compile_and_sim_time[i].pop(0)
            i += 1
    sim_averages = [sum(result) / n_sim for result in sim_time]
    compile_and_sim_averages = [sum(result) / n_sim for result in compile_and_sim_time]
    print("==========================================")
    print("diff_N:", diff_N)
    print("diff_dt:", diff_dt)
    print("n_sim:", n_sim)
    print("Simulation results:\n", sim_time, "\n", sim_averages)
    print("Compilation and Simulation results:\n", compile_and_sim_time, "\n", compile_and_sim_averages)
    print("==========================================")
