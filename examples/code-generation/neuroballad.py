#!/usr/bin/env python

"""
Neuroballad circuit class and components for simplifying Neurokernel/Neurodriver
workflow.
"""

from __future__ import absolute_import

import inspect
import os
import pickle
import subprocess
from shutil import copyfile

import h5py
import matplotlib as mpl
import networkx as nx
import numpy as np

mpl.use('agg')

class NeuroballadExecutor(object):
    def __init__(self, config = {}):
        self.path = 'neuroballad_execute.py'

def get_neuroballad_path():
    return os.path.dirname(\
    os.path.abspath(inspect.getsourcefile(NeuroballadExecutor)))

def NeuroballadModelGenerator(model_name, arg_names):
    def __init__(self, **kwargs):
        BaseNeuroballadModel.__init__(self, model_name)
        self.arg_names = arg_names + ['reverse']
        for key, value in kwargs.items():
            if key in arg_names:
                pass
            else:
                print("Warning: ", key, 'is not in the default parameters for', model_name, ':', ', '.join(arg_names))
            setattr(self, key, value)
    def nadd(self, G, i):
        name = 'uid' + str(i)
        node_name = self.__class__.__name__
        dict_struct = {'class': node_name}
        for key in self.arg_names:
            try:
                dict_struct[key] = getattr(self, key)
            except:
                pass
                # Fill in better error handling here
                # print('Required parameter',key,'not found in the definition.')
        G.add_node(name, **dict_struct)
        return G
    return type(model_name, (BaseNeuroballadModel,),{"__init__": __init__, "nadd": nadd})

class BaseNeuroballadModel(object):
    def __init__(self, my_type):
        self._type = my_type
        
def populate_models():
    from importlib import import_module
    import neurokernel
    import inspect
    import os
    nk_path = inspect.getfile(neurokernel)
    ndcomponents_path = os.path.join(os.path.join(os.path.dirname(nk_path),'LPU'),'NDComponents')
    comp_types = [f.path for f in os.scandir(ndcomponents_path) if f.is_dir() and 'Models' in os.path.basename(f.path)]    
    for i in comp_types:
        models_path_py = '.'.join(i.split('/')[-4:])
        model_paths = [f.path for f in os.scandir(i) if not f.is_dir() and 'Base' not in os.path.basename(f.path)  and '__init__' not in os.path.basename(f.path)  and '.py' in os.path.basename(f.path)]
        for p in model_paths:
            model_name = os.path.basename(p).split('.')[0]
            from_path = models_path_py + '.' + model_name
            mod = import_module(from_path)
            model_class = getattr(mod, model_name)
            if hasattr(model_class, 'params'):
                params = model_class.params
            else:
                params = []
            if model_name not in globals():
                globals()[model_name] = NeuroballadModelGenerator(model_name, params)
            else:
                print(model_name,'has been already defined in workspace.')
                
class Circuit(object):
    """
    Create a Neuroballad circuit.

    Basic Example
    --------
    >>> from neuroballad import * #Import Neuroballad
    >>> C.add([0, 2, 4], HodgkinHuxley()) #Create three neurons
    >>> C.add([1, 3, 5], AlphaSynapse()) #Create three synapses
    >>> C.join([[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]]) #Join nodes together
    >>> C_in_a = InIStep(0, 40., 0.25, 0.50) #Create current input for node 0
    >>> C_in_b = InIStep(2, 40., 0.50, 0.75) #Create current input for node 2
    >>> C_in_c = InIStep(4, 40., 0.75, 0.50) #Create current input for node 4
    >>> C.sim(1., 1e-4, [C_in_a, C_in_b, C_in_c]) #Use the inputs and simulate
    """
    def __init__(self, name = ''):
        self.G = nx.MultiDiGraph() #Neurokernel graph definition
        self.results = {} #Simulation results
        self.ids = [] #Graph ID's
        self.tracked_variables = [] #Observable variables in circuit
        self.t_duration = 0
        self.t_step = 0
        self.inputs = []
        self.outputs = []
        self.experimentConfig = []
        self.ICs = []
        self.name = name

    def add(self, name, neuron):
        """
        Loops through a list of ID inputs and adds corresponding components
        of a specific type to the circuit.

        Example
        --------
        >>> C.add([1, 2, 3], HodgkinHuxley())
        """
        if neuron.ElementClass == "input":
            self.experimentConfig.append(neuron.addToExperiment)
        else:
            for i in name:
                if (i in self.ids):
                    raise ValueError('Don''t use the same ID for multiple neurons!')
            for i in name:
                self.G = neuron.nadd(self.G, i)
                self.ids.append(i)
    def get_new_id(self):
        """
        Densely connects two arrays of circuit ID's.

        Example
        --------
        >>> C.dense_connect_via(cluster_a, cluster_b)
        """
        if self.ids == []:
            return 0
        else:
            return int(np.max(self.ids)+1)
        #return next(filter(set(self.ids).__contains__, \
        #            itertools.count(0)))
    def add_cluster(self, number, neuron):
        """
        Creates a number of components of a specific type and returns their
        ID's.

        Example
        --------
        >>> id_list = C.add_cluster(256, HodgkinHuxley())
        """
        cluster_inds = []
        for i in range(number):
            i_toadd = self.get_new_id()
            self.G = neuron.nadd(self.G, i_toadd)
            self.ids.append(i_toadd)
            cluster_inds.append(i_toadd)
        return cluster_inds
    def dense_connect_via(self, in_array_a, in_array_b, neuron, delay = 0.0, via = '', tag = 0,
                          debug = 0):
        """
        Densely connects two arrays of circuit ID's, creating a layer of unique
        components of a specified type in between.

        Example
        --------
        >>> C.dense_join_via(cluster_a, cluster_b, AlphaSynapse())
        """
        for i in in_array_a:
            for j in in_array_b:
                i_toadd = self.get_new_id()
                if debug==1:
                    print('Added neuron ID: ' + str(i_toadd))
                self.G = neuron.nadd(self.G, i_toadd)
                self.ids.append(i_toadd)
                self.join([[i, i_toadd], [i_toadd, j]], delay = delay, via=via, tag = tag)
    def dense_connect(self, in_array_a, in_array_b, delay = 0.0):
        """
        Densely connects two arrays of circuit ID's.

        Example
        --------
        >>> C.dense_connect_via(cluster_a, cluster_b)
        """
        for i in in_array_a:
            for j in in_array_b:
                self.join([[i, j]], delay = delay)
    def dense_join(self, in_array_a, in_array_b, in_array_c, delay = 0.0):
        """
        Densely connects two arrays of circuit ID's, using a third array as the
        matrix of components that connects the two.

        Example
        --------
        >>> C.dense_join_via(cluster_a, cluster_b, cluster_c)
        """
        k = 0
        in_array_c = in_array_c.flatten()
        for i in in_array_a:
            for j in in_array_b:
                self.join([[i, in_array_c[k]], [in_array_c[k], j]]
                          , delay = delay)
                k += 1
    def join(self, in_array, delay = 0.0, via = '', tag = 0):
        """
        Processes an edge list and adds the edges to the circuit.

        Example
        --------
        >>> C.add([0, 2, 4], HodgkinHuxley()) #Create three neurons
        >>> C.add([1, 3, 5], AlphaSynapse()) #Create three synapses
        >>> C.join([[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]])
        """
        in_array = np.array(in_array)
        #print(in_array)
        for i in range(in_array.shape[0]):
            if via == '':
                self.G.add_edge('uid' + str(in_array[i,0]),
                                'uid' + str(in_array[i,1]), delay = delay, tag = tag)
            else:
                self.G.add_edge('uid' + str(in_array[i,0]),
                                'uid' + str(in_array[i,1]), delay = delay,
                                via = via, tag = tag)
    def fit(self, inputs):
        """
        Attempts to find parameters to fit a certain curve to the output.
        Not implemented at this time.
        """
        pass
    def compile(self, model_output_name = 'neuroballad_temp_model.gexf.gz'):
        """
        Writes the current circuit to a file.

        Example
        --------
        >>> C.compile(model_output_name = 'example_circuit.gexf.gz')
        """
        nx.write_gexf(self.G, model_output_name)
    def load_last(self, file_name = 'neuroballad_temp_model.gexf.gz'):
        """
        Loads the latest executed circuit in the directory.

        Example
        --------
        >>> C.load_last()
        """
        self.G = nx.read_gexf()
        self.ids = []
        for i in self.G.nodes():
            self.ids.append(int(i[3:]))
    def save(self, file_name = 'neuroballad_temp_model.gexf.gz'):
        """
        Saves the current circuit to a file.

        Example
        --------
        >>> C.save(file_name = 'my_circuit.gexf.gz')
        """
        nx.write_gexf(self.G, file_name)
    def sim(self, t_duration, t_step, in_list = None, record = ['V', 'spike_state', 'I']):
        """
        Simulates the circuit for a set amount of time, with a fixed temporal
        step size and a list of inputs.

        Example
        --------
        >>> C.sim(1., 1e-4, InIStep(0, 10., 1., 2.))
        """
        self.t_duration = t_duration
        self.t_step = t_step
        if in_list is None:
            in_list = self.experimentConfig
        run_parameters = [self.t_duration, self.t_step]
        with open('run_parameters.pickle', 'wb') as f:
            pickle.dump(run_parameters, f, protocol=pickle.HIGHEST_PROTOCOL)
        nx.write_gexf(self.G, 'neuroballad_temp_model.gexf.gz')
        Nt = int(t_duration/t_step)
        t  = np.arange(0, t_step*Nt, t_step)
        uids = []
        for i in in_list:
            uids.append('uid' + str(i.node_id))
        input_vars = []
        for i in in_list:
            input_vars.append(i.var)
        input_vars = list(set(input_vars))
        uids = np.array(list(set(uids)), dtype = 'S')
        Is = {}
        Inodes = {}
        for i in input_vars:
            Inodes[i] = []
        for i in in_list:
            Inodes[i.var].append('uid' + str(i.node_id))
        for i in input_vars:
            Inodes[i] = np.array(list(set(Inodes[i])), dtype = 'S')
        for i in input_vars:
            Is[i] = np.zeros((Nt, len(Inodes[i])), dtype=np.float64)
        # I = np.zeros((Nt, len(uids)), dtype=np.float64)
        file_name = 'neuroballad_temp_model_input.h5'
        for i in in_list:
            Is[i.var] = i.add(Inodes[i.var], Is[i.var], t)
        with h5py.File(file_name, 'w') as f:
            for i in input_vars:
                print(i + '/uids')
                f.create_dataset(i + '/uids', data=Inodes[i])
                f.create_dataset(i + '/data', (Nt, len(Inodes[i])),
                                dtype=np.float64,
                                data=Is[i])
        recorders = []
        for i in record:
            recorders.append((i,None))
        with open('record_parameters.pickle', 'wb') as f:
            pickle.dump(recorders, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(recorders)
        if os.path.isfile('neuroballad_execute.py'):
            subprocess.call(['python','neuroballad_execute.py'])
        else:
            copyfile(get_neuroballad_path() + '/neuroballad_execute.py',\
                     'neuroballad_execute.py')
            subprocess.call(['python','neuroballad_execute.py'])
    def collect_results(self):
        """
        Collects the latest results from the executor. Useful when loading
        a set of results after execution.

        Example
        --------
        >>> C.collect_results()
        """
        import neurokernel.LPU.utils.visualizer as vis
        self.V = vis.visualizer()
        self.V.add_LPU('neuroballad_temp_model_output.h5',
                  gexf_file = 'neuroballad_temp_model.gexf.gz',LPU = 'lpu')
        # print([self.V._uids['lpu']['V']])
    def visualize_video(self, name, config = {}, visualization_variable = 'V',
                        out_name = 'test.avi'):
        """
        Visualizes all ID's using a set visualization variable over time,
        saving them to a video file.

        Example
        --------
        >>> C.visualize_video([0, 2, 4], out_name='visualization.avi')
        """
        uids = []
        if config == {}:
            config = {'variable': visualization_variable, 'type': 'waveform',
                      'uids': [self.V._uids['lpu'][visualization_variable]]}
        for i in name:
            print(i)
            uids.append('uid' + str(i))
        config['uids'] = uids
        self.V.codec = 'mpeg4'
        self.V.add_plot(config, 'lpu')
        self.V.update_interval = 1e-4
        self.V.out_filename = out_name
        self.V.run()
    def visualize_circuit(self, prog = 'dot' , splines = 'line'):
        print(self.ids)
        styles = {
            'graph': {
                'label': self.name,
                #'fontname': 'LM Roman 10',
                'fontsize': '16',
                'fontcolor': 'black',
                #'bgcolor': '#333333',
                #'rankdir': 'LR',
                'splines': splines,
                'model': 'circuit',
                'size': '250,250',
                'overlap': 'false',
            },
            'nodes': {
                #'fontname': 'LM Roman 10',
                'shape': 'box',
                'fontcolor': 'black',
                'color': 'black',
                'style': 'rounded',
                'fillcolor': '#006699',
                #'nodesep': '1.5',
            },
            'edges': {
                'style': 'solid',
                'color': 'black',
                'arrowhead': 'open',
                'arrowsize': '0.5',
                'fontname': 'Courier',
                'fontsize': '12',
                'fontcolor': 'black',
                'splines': 'ortho',
                'concentrate': 'false',
            }
        }
        #G = nx.read_gexf('neuroballad_temp_model.gexf.gz')
        G = self.G
        # G.remove_nodes_from(nx.isolates(G))
        mapping = {}
        node_types = set()
        for n,d in G.nodes(data=True):
            node_types.add( d['name'].rstrip('1234567890') )
        node_nos = dict.fromkeys(node_types, 1)
        for n,d in G.nodes(data=True):
            node_type = d['name'].rstrip('1234567890')
            mapping[n] = d['name'].rstrip('1234567890') + str(node_nos[node_type])
            node_nos[node_type] += 1
        G = nx.relabel_nodes(G,mapping)
        A = nx.drawing.nx_agraph.to_agraph(G)
        #A.graph_attr['fontname']= 'LM Roman 10'
        #A.graph_attr['splines'] = 'ortho'
        #A.graph_attr['bgcolor'] = '#333333'
        A.graph_attr.update(styles['graph'])
        A.write('file.dot')
        for i in A.edges():
            e=A.get_edge(i[0],i[1])
            #e.attr['splines'] = 'ortho'
            e.attr.update(styles['edges'])
            if i[0][:-1] == 'Repressor':
                e.attr['arrowhead'] = 'tee'
        for i in A.nodes():
            n=A.get_node(i)
            print(n)
            #n.attr['shape'] = 'box'
            n.attr.update(styles['nodes'])
        A.layout(prog=prog)
        A.draw('neuroballad_temp_circuit.svg')
        A.draw('neuroballad_temp_circuit.eps')


### Component Definitions

class HodgkinHuxley(object):
    ElementClass = 'neuron'
    def __init__(self, name = "", n = 0.5, m = 0.5, h = 0.5, initV = -60.0):
        self.n = n
        self.m = m
        self.h = h
        self.initV = initV
        self.name = name
    def nadd(self, G, i):
        name = 'uid' + str(i)
        node_name = self.name
        if node_name == "":
            node_name = 'HodgkinHuxley' + str(i)
        G.add_node(name, **{'class': 'HodgkinHuxley',
         'name': node_name, 'm': self.m,
         'h': self.h})
        attrs = {name: {'n': self.n}}
        nx.set_node_attributes(G, attrs)
        return G

class ConnorStevens(object):
    ElementClass = 'neuron'
    def __init__(self, name = "", n = 0.0, m = 0.0, h = 1.0, a = 0., b = 0.):
        self.n = n
        self.m = m
        self.h = h
        self.a = a
        self.b = b
    def nadd(self, G, i):
        name = 'uid' + str(i)
        G.add_node(name, **{'class': 'ConnorStevens',
         'name': 'ConnorStevens' + str(i), 'm': self.m,
         'h': self.h, 'a': self.a, 'b': self.b})
        attrs = {name: {'n': self.n}}
        nx.set_node_attributes(G, attrs)
        return G

class LeakyIAF(object):
    ElementClass = 'neuron'
    def __init__(self, name = "", resting_potential = -67.5489770451,
                 reset_potential = -67.5489770451, threshold = -25.1355161007,
                 capacitance = 0.0669810502993,
                 resistance = 1002.445570216,
                 initV = 10001.):
        self.resting_potential = (resting_potential)
        self.reset_potential = (reset_potential)
        self.threshold = (threshold)
        self.capacitance = (capacitance)
        self.resistance = (resistance)
        self.name = name
        if initV > threshold:
            self.initV = np.random.uniform(reset_potential, threshold)
        else:
            self.initV = initV

    def nadd(self, G, i):
        name = 'uid' + str(i)
        G.add_node(name, **{'class': 'LeakyIAF', 'name': 'LeakyIAF' + str(i),
         'resting_potential': self.resting_potential, 'initV': self.initV,
         'reset_potential': self.reset_potential, 'threshold': self.threshold,
         'capacitance': self.capacitance, 'resistance': self.resistance})
        return G

class myLeakyIAF(object):
    ElementClass = 'neuron'
    def __init__(self, name = "", resting_potential = -67.5489770451,
                 reset_potential = -67.5489770451, threshold = -25.1355161007,
                 capacitance = 0.0669810502993,
                 resistance = 1002.445570216,
                 initV = 10001.):
        self.resting_potential = (resting_potential)
        self.reset_potential = (reset_potential)
        self.threshold = (threshold)
        self.capacitance = (capacitance)
        self.resistance = (resistance)
        self.name = name
        if initV > threshold:
            self.initV = np.random.uniform(reset_potential, threshold)
        else:
            self.initV = initV

    def nadd(self, G, i):
        name = 'uid' + str(i)
        G.add_node(name, **{'class': 'myLeakyIAF', 'name': 'myLeakyIAF' + str(i),
         'resting_potential': self.resting_potential, 'initV': self.initV,
         'reset_potential': self.reset_potential, 'threshold': self.threshold,
         'capacitance': self.capacitance, 'resistance': self.resistance})
        return G

class DopamineLeakyIAF(object):
    ElementClass = 'neuron'
    def __init__(self, name = "", resting_potential = -67.5489770451,
                 reset_potential = -67.5489770451, threshold = -25.1355161007,
                 capacitance = 0.0669810502993,
                 resistance = 1002.445570216,
                 initV = 10001.):
        self.resting_potential = (resting_potential)
        self.reset_potential = (reset_potential)
        self.threshold = (threshold)
        self.capacitance = (capacitance)
        self.resistance = (resistance)
        self.name = name
        if initV > threshold:
            self.initV = np.random.uniform(-60.0,-25.0)
        else:
            self.initV = initV
    def nadd(self, G, i):
        name = 'uid' + str(i)
        G.add_node(name, **{'class': 'DopamineLeakyIAF',
         'name': 'DopamineLeakyIAF' + str(i),
         'resting_potential': self.resting_potential, 'initV': self.initV,
         'reset_potential': self.reset_potential, 'threshold': self.threshold,
         'capacitance': self.capacitance, 'resistance': self.resistance})
        return G

class AlphaSynapse(object):
    ElementClass = 'synapse'
    def __init__(self, name = "", ar = 1.1*1e2, ad = 1.9*1e2, reverse = 65.0,
                 gmax = 3*1e-6):
        self.ar = ar
        self.ad = ad
        self.reverse = reverse
        self.gmax = gmax
        self.name = name
    def nadd(self, G, i):
        name = 'uid' + str(i)
        G.add_node(name, **{'class': 'AlphaSynapse',
         'name': 'AlphaSynapse' + str(i), 'ar': self.ar, 'ad': self.ad,
         'reverse': self.reverse, 'gmax': self.gmax, 'circuit': 'local' })
        return G

class myAlphaSynapse(object):
    ElementClass = 'synapse'
    def __init__(self, name = "", ar = 1.1, ad = 1.9, reverse = 65.0,
                 gmax = 3*1e-6):
        self.ar = ar
        self.ad = ad
        self.reverse = reverse
        self.gmax = gmax
        self.name = name
    def nadd(self, G, i):
        name = 'uid' + str(i)
        G.add_node(name, **{'class': 'myAlphaSynapse',
         'name': 'myAlphaSynapse' + str(i), 'ar': self.ar, 'ad': self.ad,
         'reverse': self.reverse, 'gmax': self.gmax, 'circuit': 'local' })
        return G

class DopamineAlphaSynapse(object):
    ElementClass = 'synapse'
    def __init__(self, name = "", ar = 1.1*1e2, ad = 1.9*1e3, reverse = 65.0,
                 gmax = 3*1e-6):
        self.ar = ar
        self.ad = ad
        self.reverse = reverse
        self.gmax = gmax
        self.name = name
    def nadd(self, G, i):
        name = 'uid' + str(i)
        G.add_node(name, **{'class': 'DopamineAlphaSynapse',
         'name': 'DopamineAlphaSynapse' + str(i), 'ar': self.ar, 'ad': self.ad,
         'reverse': self.reverse, 'gmax': self.gmax, 'circuit': 'local' })
        return G

class PowerGPotGPot(object):
    ElementClass = 'synapse'
    def __init__(self, name = "", gmax = 0.4, threshold = -55.0, slope = 0.02,
                 power = 1.0, saturation = 0.4, reverse = 0.0):
        self.gmax = gmax
        self.threshold = float(threshold)
        self.slope = slope
        self.power = power
        self.saturation = saturation
        self.reverse = reverse
    def nadd(self, G, i):
        name = 'uid' + str(i)
        G.add_node(name, **{'class': 'PowerGPotGPot',
         'name': 'PowerGPotGPot' + str(i), 'gmax': self.gmax,
         'threshold': self.threshold, 'slope': self.slope, 'power': self.power,
         'saturation': self.saturation, 'reverse': self.reverse })
        return G

class MorrisLecar(object):
    ElementClass = 'neuron'
    def __init__(self, name = "", V1 = -20., V2 = 50.0, V3 = -40.,
                 V4 = 20.0, phi = 0.4, offset= 0., V_L = -40., V_Ca = 120.,
                 V_K = -80., g_L = 3., g_Ca = 4., g_K = 16., initV = -46.080,
                 initn = 0.3525):
        self.V1 = V1
        self.V2 = V2
        self.V3 = V3
        self.V4 = V4
        self.phi = phi
        self.offset = offset
        self.V_L = V_L
        self.V_Ca = V_Ca
        self.V_K = V_K
        self.g_L = g_L
        self.g_Ca = g_Ca
        self.g_K = g_K
        self.initV = initV
        self.initn = initn
    def nadd(self, G, i):
        name = 'uid' + str(i)
        G.add_node(name, **{'class': 'MorrisLecar',
         'name': 'MorrisLecar' + str(i), 'V1': self.V1, 'V2': self.V2,
         'V3': self.V3, 'V4': self.V4, 'phi': self.phi, 'offset': self.offset,
         'V_L': self.V_L, 'V_Ca': self.V_Ca, 'V_K': self.V_K, 'g_L': self.g_L,
         'g_Ca': self.g_Ca, 'g_K': self.g_K, 'initV': self.initV,
         'initn': self.initn})
        return G

class DopamineModulatedAlphaSynapse(object):
    ElementClass = 'synapse'
    def __init__(self, name = "", ar = 1.1*1e2, ad = 1.9*1e3, reverse = 65.0,
                 gmax = 3*1e-6):
        self.ar = ar
        self.ad = ad
        self.reverse = reverse
        self.gmax = gmax
        self.name = name
    def nadd(self, G, i):
        name = 'uid' + str(i)
        G.add_node(name, **{'class': 'DopamineModulatedAlphaSynapse',
         'name': 'DopamineModulatedAlphaSynapse' + str(i), 'ar': self.ar,
         'ad': self.ad, 'reverse': self.reverse, 'gmax': self.gmax,
         'circuit': 'local' })
        return G

class Activator(object):
    ElementClass = 'abstract'
    def __init__(self, name = "", beta = 1.0, K = 1.0, n = 1.0):
        self.beta = beta
        self.K = K
        self.n = n
        self.name = name
    def nadd(self, G, i):
        name = 'uid' + str(i)
        G.add_node(name, **{'class': 'Activator',
         'name': 'Activator' + str(i), 'beta': self.beta,
         'K': self.K,
         'circuit': 'local' })
        attrs = {name: {'n': self.n}}
        nx.set_node_attributes(G, attrs)
        return G

class Repressor(object):
    ElementClass = 'abstract'
    def __init__(self, name = "", beta = 1.0, K = 1.0, n = 1.0):
        self.beta = beta
        self.K = K
        self.n = n
        self.name = name
    def nadd(self, G, i):
        name = 'uid' + str(i)
        G.add_node(name, **{'class': 'Repressor',
         'name': 'Repressor' + str(i), 'beta': self.beta,
         'K': self.K,
         'circuit': 'local' })
        attrs = {name: {'n': self.n}}
        nx.set_node_attributes(G, attrs)
        return G

class Integrator(object):
    ElementClass = 'abstract'
    def __init__(self, name = "", gamma = 0.0):
        self.gamma = gamma
        self.name = name
    def nadd(self, G, i):
        name = 'uid' + str(i)
        G.add_node(name, **{'class': 'Integrator',
         'name': 'Integrator' + str(i), 'gamma': self.gamma,
         'circuit': 'local' })
        return G

class CurrentModulator(object):
    ElementClass = 'abstract'
    def __init__(self, name = "", A = 1.0, shift = 0.0):
        self.A = A
        self.shift = shift
        self.name = name
    def nadd(self, G, i):
        name = 'uid' + str(i)
        G.add_node(name, **{'class': 'CurrentModulator',
         'name': 'CurrentModulator' + str(i), 'A': self.A, 'shift': self.shift,
         'circuit': 'local' })
        return G

class Threshold(object):
    ElementClass = 'abstract'
    def __init__(self, name = "", threshold_value = 1.0, threshold_mode = 0.0):
        self.threshold_value = threshold_value
        self.threshold_mode = threshold_mode
        self.name = name
    def nadd(self, G, i):
        name = 'uid' + str(i)
        G.add_node(name, **{'class': 'Threshold',
         'name': 'Threshold' + str(i),
         'threshold_value': self.threshold_value,
         'threshold_mode': self.threshold_mode,
         'circuit': 'local' })
        return G

class Segev(object):
    ElementClass = 'neuron'
    def __init__(self, name = "", C = 0.0669810502993, R = 1002.445570216, V_leak = 1.0):
        self.C = C
        self.R = R
        self.V_leak = V_leak
        self.name = name
    def nadd(self, G, i):
        name = 'uid' + str(i)
        G.add_node(name, **{'class': 'Segev',
         'name': 'Segev' + str(i),
         'C': self.C,
         'R': self.R,
         'V_leak': self.V_leak,
         'circuit': 'local' })
        return G

class Chemical(object):
    ElementClass = 'synapse'
    def __init__(self, name = "",  reverse = 65.0, g_max = 6. * (10. ** -1.), K = -4.3944, V_eq = -24.0, V_range = 36.0):
        self.g_max = g_max
        self.K = K
        self.V_eq = V_eq
        self.V_range = V_range
        self.reverse = reverse
        self.name = name
    def nadd(self, G, i):
        name = 'uid' + str(i)
        G.add_node(name, **{'class': 'Chemical',
         'name': 'Chemical' + str(i),
         'g_max': self.g_max,
         'K': self.K,
         'V_eq': self.V_eq,
         'V_range': self.V_range,
         'reverse': self.reverse,
         'circuit': 'local' })
        return G

class Resistor(object):
    ElementClass = 'abstract'
    def __init__(self, name = "", R = 1.0):
        self.R = R
        self.name = name
    def nadd(self, G, i):
        name = 'uid' + str(i)
        G.add_node(name, **{'class': 'Resistor',
         'name': 'Resistor' + str(i),
         'R': self.R,
         'circuit': 'local' })
        return G

class Aggregator(object):
    ElementClass = 'abstract'
    def __init__(self, name = ""):
        self.name = name
    def nadd(self, G, i):
        name = 'uid' + str(i)
        G.add_node(name, **{'class': 'Aggregator',
         'name': 'Aggregator' + str(i),
         'circuit': 'local' })
        return G

class AuditoryTransducer(object):
    ElementClass = 'abstract'
    def __init__(self, name = "",
                 M = 1.0,
                 K_ho = 1.0,
                 K_gs = 1.0,
                 K_aj = 1.0,
                 N = 1.0,
                 D = 1.0,
                 delta_G = 1.0,
                 k_b = 1.0,
                 T = 1.0,
                 delta = 1.0,
                 S = 1.0,
                 F_max = 1.0,
                 lambd = 1.0,
                 lambda_a = 1.0,
                 P_zero_rest = 0.5):
        self.M = M
        self.K_ho = K_ho
        self.K_gs = K_gs
        self.K_aj = K_aj
        self.N = N
        self.D = D
        self.delta_G = delta_G
        self.k_b = k_b
        self.T = T
        self.delta = delta
        self.S = S
        self.F_max = F_max
        self.lambd = lambd
        self.lambda_a = lambda_a
        self.P_zero_rest = P_zero_rest
    def nadd(self, G, i):
        name = 'uid' + str(i)
        G.add_node(name, **{'class': 'AuditoryTransducer',
         'name': 'AuditoryTransducer' + str(i),
         'M': self.M,
         'K_ho': self.K_ho,
         'K_gs': self.K_gs,
         'K_aj': self.K_aj,
         'D': self.D,
         'delta_G': self.delta_G,
         'k_b': self.k_b,
         'T': self.T,
         'delta': self.delta,
         'S': self.S,
         'F_max': self.F_max,
         'lambda': self.lambd,
         'lambda_a': self.lambda_a,
         'P_zero_rest': self.P_zero_rest,
         'circuit': 'local' })
        attrs = {name: {'n': self.n}}
        nx.set_node_attributes(G, attrs)
        return G

class OutPort(object):
    ElementClass = 'port'
    def __init__(self, name = "", port_type = 'gpot', lpu = 'lpu'):
        self.port_type = port_type
        self.lpu = lpu
        self.name = name
    def nadd(self, G, i):
        name = 'uid' + str(i)
        G.add_node(name, **{'class': 'Port', 'name': 'Port' + str(i),
         'port_type': self.port_type, 'port_io': 'out',
         'selector': '/%s/out/%s/%s' % (self.lpu, self.port_type, str(i))})
        return G

class InPort(object):
    ElementClass = 'port'
    def __init__(self, name = "", port_type = 'spike', lpu = 'lpu'):
        self.port_type = port_type
        self.lpu = lpu
        self.name = name
    def nadd(self, G, i):
        name = 'uid' + str(i)
        G.add_node(name, **{'class': 'Port', 'name': 'Port' + str(i),
         'port_type': self.port_type, 'port_io': 'in',
         'selector': '/%s/in/%s/%s' % (self.lpu, self.port_type, str(i))})
        return G
### Input Processors

class InIBoxcar(object):
    ElementClass = 'input'
    def __init__(self, node_id, I_val, t_start, t_end, var = 'I'):
        self.node_id = node_id
        self.I_val = I_val
        self.t_start = t_start
        self.t_end = t_end
        self.var = var
        a = {}
        a['name'] = 'InIBoxcar'
        a['node_id'] = node_id
        a['I_val'] = I_val
        a['t_start'] = t_start
        a['t_end'] = t_end
        self.params = a
    def add(self, uids, I, t):
        step_range = [self.t_start, self.t_end]
        step_intensity = self.I_val
        uids = [i.decode("utf-8") for i in uids]
        I[np.logical_and(t>step_range[0], t<step_range[1]),
        np.where([i == ('uid' + str(self.node_id)) for i in uids])] += step_intensity
        return I
    def addToExperiment(self):
        return self.params

class InIStep(InIBoxcar):
    ElementClass = 'input'
    def __init__(self, node_id, I_val, t_start, t_end, var = 'I'):
        InIBoxcar.__init__(self, node_id, I_val, t_start, t_end, var = var)

class InSpike(object):
    ElementClass = 'input'
    def __init__(self, node_id, density, var = 'I'):
        self.node_id = node_id
        self.density = density
        self.var = var
        a = {}
        a['name'] = 'InSpike'
        a['node_id'] = node_id
        a['density'] = density
        self.params = a
    def add(self, uids, I, t):
        uids = [i.decode("utf-8") for i in uids]
        I[np.nonzero((t - np.round(t / self.density) * self.density) == 0)[0],
        np.where([i == ('uid' + str(self.node_id)) for i in uids])] = 1.
        return I
    def addToExperiment(self):
        return self.params

class InIGaussianNoise(object):
    ElementClass = 'input'
    def __init__(self, node_id, mean, std, t_start, t_end, var='I'):
        self.node_id = node_id
        self.mean = mean
        self.std = std
        self.t_start = t_start
        self.t_end = t_end
        self.var = var
        a = {}
        a['name'] = 'InIGaussianNoise'
        a['node_id'] = node_id
        a['mean'] = mean
        a['std'] = std
        a['t_start'] = t_start
        a['t_end'] = t_end
        self.params = a
    def add(self, uids, I, t):
        step_range = [self.t_start, self.t_end]
        uids = [i.decode("utf-8") for i in uids]
        I[np.logical_and(t>step_range[0], t<step_range[1]),
        np.where([i == ('uid' + str(self.node_id)) for i in uids])] += self.mean + self.std*\
        np.array(np.random.randn(len(np.where(np.logical_and(t>step_range[0], \
        t<step_range[1])))))
        return I
    def addToExperiment(self):
        return self.params

class InISinusoidal(object):
    ElementClass = 'input'
    def __init__(self, 
                 node_id, 
                 amplitude, 
                 frequency, 
                 t_start, 
                 t_end, 
                 mean = 0, 
                 shift = 0., 
                 frequency_sweep = 0.0, 
                 frequency_sweep_frequency = 1.0, 
                 threshold_active = 0, 
                 threshold_value = 0.0,
                 var = 'I'):
        self.node_id = node_id
        self.amplitude = amplitude
        self.frequency = frequency
        self.mean = mean
        self.t_start = t_start
        self.t_end = t_end
        self.shift = shift
        self.threshold_active = threshold_active
        self.threshold_value = threshold_value
        self.frequency_sweep_frequency = frequency_sweep_frequency
        self.frequency_sweep = frequency_sweep
        self.var = var
        a = {}
        a['name'] = 'InISinusoidal'
        a['node_id'] = node_id
        a['amplitude'] = amplitude
        a['frequency'] = frequency
        a['t_start'] = t_start
        a['t_end'] = t_end
        a['mean'] = mean
        a['shift'] = shift
        a['frequency_sweep'] = frequency_sweep
        a['frequency_sweep_frequency'] = frequency_sweep_frequency
        a['threshold_active'] = threshold_active
        a['threshold_value'] = threshold_value
        self.params = a
    def add(self, uids, I, t):
        step_range = [self.t_start, self.t_end]
        uids = [i.decode("utf-8") for i in uids]
        sin_wave = np.sin(2 * np.pi * t * (self.frequency + self.frequency_sweep * np.sin(2 * np.pi * t * self.frequency_sweep_frequency)) + self.shift)
        values_to_add = self.mean + self.amplitude * \
                sin_wave[np.logical_and(t>step_range[0], t<step_range[1])]
        if self.threshold_active>0:
            values_to_add[values_to_add>self.threshold_value] = np.max(values_to_add)
            values_to_add[values_to_add<=self.threshold_value] = np.min(values_to_add)
        I[np.logical_and(t>step_range[0], t<step_range[1]),
        np.where([i == ('uid' + str(self.node_id)) for i in uids])] += values_to_add
        return I
    def addToExperiment(self):
        return self.params
