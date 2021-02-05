
from collections import OrderedDict
import uuid

import h5py
import numpy as np
import networkx as nx

import pycuda.driver as cuda
import pycuda.gpuarray as garray
from neurokernel.LPU.InputProcessors.ArrayInputProcessor import ArrayInputProcessor

class PresynapticInputProcessor(ArrayInputProcessor):
    """
    Allow users to specify inputs to neurons from presynaptic variables such as
    voltage and spike_state, and automatically add user-defined synapses.

    Parameters
    ----------
    input: dict
           A dictionary with the following structure:
           {var1: {'uids': ['comp1', 'comp2' ... ],
                   'data': np.ndarray,
                   'components': {'model': 'AlphaSynpase', 'gmax': 10.0,
                                  'ar': 0.19, 'ad': 0.16}
                  },
            var2: {'uids': ['comp3', 'comp4' ... ],
                   'data': np.ndarray,
                   'components': [{'model': 'SynapseAMPA', 'gmax': 10.0, 'st': 4.0},
                                  {'model': 'SynapseGABA', 'gmax': 8.0, 'st': 4.0},
                                  ...]
                  },
            ...}
           where in each key/value pair, the key specifies the varialbe to be
           inject input to, value['uids'] is a list that specifies the uids of
           components to be injected to, and value['data'] is an np.ndarray
           of shape (T, N) that specifies the input to each of the N components
           at every time step for a total of T steps, and the ordering the data in
           each column should be the same as the order in value['uids'].
           For 'spike_state', in addition to provide 0 (no spike) or 1 (spike)
           for each time step, spike time and the index of neuron that spiked
           can be provided by value['data']['time'] and value['data']['index'],
           respectively. See examples for more details.

           The input can also be programmatically loaded by other methods.
    mode:     int
              mode in BaseInputProcessor
              0: default, when the file does not cover the entire simulation,
                 inputs default to 0.
              1: when the file does not cover the entire simulation,
                 inputs defaults to the last state.
    cache_length: int
                  length (number of steps) of cache on GPU memory to preload
                  inputs.

    Examples
    --------

    """
    def add_inputs(self, inputs):
        for var, g in inputs.items():
            self.add_input(var, g['uids'], g['data'], g['components'])

    def add_input(self, var, uids, data, components):
        new_uids = []
        for i, uid in enumerate(uids):
            if not uid in self._additional_graph.nodes:
                self._additional_graph.add_node(uid)
            new_uid = 'synaptic_input_to_{}_{}'.format(uid, str(uuid.uuid4()).split('-')[0])
            params = components if isinstance(components, dict) else components[i]
            self._additional_graph.add_node(new_uid, **params)
            new_uids.append(new_uid)
            self._additional_graph.add_edge(new_uid, uid)

        if var in self.variables:
            if var == 'spike_state':
                if isinstance(data, dict):
                    assert issubclass(data['index'].dtype.type, np.integer), \
                           'spike index array must be integer dtype'
                    self.data[var]['time'] = np.concatenate(
                            self.data[var]['time'], data['time'])
                    self.data[var]['index'] = np.concatenate(
                            self.data[var]['index'],
                            data['index']+len(self.variables[var]['uids']))
                else:
                    self.data[var] = np.hstack((self.data[var], data))
            else:
                self.data[var] = np.hstack((self.data[var], data))
        else:
            self.data[var] = data
            if var == 'spike_state':
                if isinstance(data, dict):
                    self.spike_state_format = 'event'
                else:
                    self.spike_state_format = 'stepwise'
        self.add_variables([(var, new_uids)])

if __name__ == '__main__':
    import networkx as nx
    from neurokernel.LPU.LPU import LPU
    from neurokernel.LPU.OutputProcessors.OutputRecorder import OutputRecorder

    dur = 0.2
    dt = 1e-4
    steps = int(dur/dt)
    G = nx.MultiDiGraph()
    uid = 'synapse1'
    scale = 0.1
    ar = 0.110
    ad = 0.190
    gmax = 0.01
    G.add_node(uid, name = 'alpha_synapse',
               model = 'AlphaSynapse2',
               ar = ar,
               ad = ad,
               scale = scale,
               gmax = gmax)

    spike_times = np.array([0.038, 0.048, 0.078], np.float64)
    input_processor = ArrayInputProcessor(
        {'spike_state':
            {'uids': [uid],
             'data': {'time': spike_times,
                      'index': np.array([0]*spike_times.shape[0], np.int32)}
            }
        })
    output_processor = OutputRecorder([('g', [uid])], dur, dt)
    lpu = LPU(dt, 'obj', {'graph': G, 'kwargs': {'class_key': 'model'}},
              device = 0,
              id = 'test', input_processors = [input_processor],
              output_processors = [output_processor],
              debug = True,
              manager = False,
              print_timing = False,
              extra_comps = [ArrayInputProcessor])
    lpu.run(steps = steps)

    s = np.zeros(steps, np.double)
    t = np.arange(0, dur, dt)
    for tk in spike_times:
        s += (scale/(ar-ad) * (np.exp(-ad*(t-tk)*1000) - np.exp(-ar*(t-tk)*1000) ))*(t>tk)

    if np.abs(output_processor.output['g']['data'].reshape(-1)-s).max() < 1e-12:
        print('Test Passed')
    else:
        print('Test Failed')
