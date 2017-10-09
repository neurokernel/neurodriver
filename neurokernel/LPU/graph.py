import networkx as nx
import inspect
from NDComponents import *
from collections import OrderedDict

def get_all_subclasses(cls):
    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses

class Graph(object):
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.modelDefaults = {}

    def _str_to_model(self, model):
        if type(model) is str:
            model = model.encode('utf-8')
            modules = get_all_subclasses(NDComponent.NDComponent)
            modules = {x.__name__.encode('utf-8'): x for x in modules}
            if model in modules:
                model = modules[model]
            else:
                raise TypeError("Unsupported model type %r" % model)
        return model

    def _parse_model_kwargs(self, model, **kwargs):
        params = kwargs.pop('params', dict())
        states = kwargs.pop('states', dict())
        model = self._str_to_model(model)

        if model not in self.modelDefaults:
            self.set_model_default(model, model.params, model.states)

        assert(set(params.keys()) <= set(self.modelDefaults[model]['params']))
        assert(set(states.keys()) <= set(self.modelDefaults[model]['states']))

        attrs = {}
        for k,v in kwargs.items():
            if k in self.modelDefaults[model]['params']:
                params[k] = v
            elif k in self.modelDefaults[model]['states']:
                states[k] = v
            else:
                attrs[k] = v
        return model, params, states, attrs

    def add_neuron(self, name, model, **kwargs):
        """Add a single neuron.

        Parameters
        ----------
        name : node
            A node can be any hashable Python object except None.
        model : string or submodule of NDComponent
            Name or the Python class of a neuron model.
        params: dict
            Parameters of the neuron model.
        states: dict
            Initial values of the state variables of the neuron model.
        kwargs:
            Key/Value pairs of extra attributes. Key could be an attribute in
            params or states.

        Examples
        --------
        >>> G = Graph()
        >>> G.add_neuron(1, 'LeakyIAF')
        >>> G.add_neuron(2, 'HodgkinHuxley', states={'n':0., 'm':0., 'h':1.})
        >>> G.add_neuron(1, 'LeakyIAF', threshould=5)

        Notes
        -----
        A hashable object is one that can be used as a key in a Python
        dictionary. This includes strings, numbers, tuples of strings
        and numbers, etc.
        """
        model, params, states, attrs = self._parse_model_kwargs(model, **kwargs)

        self.graph.add_node(name,
            {'class':model, 'params':params, 'states':states},
            **attrs)

    def set_model_default(self, model, params, states):
        model = self._str_to_model(model)
        self.modelDefaults[model] = {
            'params': params.copy(),
            'states': states.copy()
        }

    def update_model_default(self, model, **kwargs):
        for k,v in kwargs.items():
            for p in ('states', 'params'):
                if k == p:
                    self.modelDefaults[model] = v
                    break
                attr = self.modelDefaults[model][p]
                if k in attr:
                    attr[k] = v
                    break

    def add_synapse(self, name, source, target, model, **kwargs):
        """Add a single synapse.

        Parameters
        ----------
        name : hashable
            A name can be any hashable Python object except None.
        source : hashable or None
            A source can be any hashable Python object except None. The hash
            value of the pre-synaptic neuron. If None, the edge between 'source'
            and 'name' will be omitted.
        target : hashable or None
            A target can be any hashable Python object except None. The hash
            value of the post-synaptic neuron. If None, the edge between
            'target' and 'name' will be omitted.
        model : string or submodule of NDComponent
            Name or the Python class of a neuron model.
        params: dict
            Parameters of the neuron model.
        states: dict
            Initial values of the state variables of the neuron model.
        kwargs:
            Key/Value pairs of extra attributes. Key could be an attribute in
            params or states.

        Examples
        --------
        >>> G = Graph()
        >>> G.add_neuron('1', 'LeakyIAF')
        >>> G.add_neuron('2', 'HodgkinHuxley', states={'n':0., 'm':0., 'h':1.})
        >>> G.add_synapse('1->2', '1', '2', 'AlphaSynapse')

        Notes
        -----
        A hashable object is one that can be used as a key in a Python
        dictionary. This includes strings, numbers, tuples of strings
        and numbers, etc.
        """
        model, params, states, attrs = self._parse_model_kwargs(model, **kwargs)

        self.graph.add_node(name,
            {'class':model, 'params':params, 'states':states},
            **attrs)

        self.graph.add_edge(source, name)
        self.graph.add_edge(name, target)

    def write_gexf(self, filename):
        graph = nx.MultiDiGraph()
        for n,d in self.graph.nodes_iter(data=True):
            data = d.copy()
            model = data.pop('class')
            data['class'] = model.__name__.encode('utf-8')
            for p in ('params', 'states'):
                r = self.modelDefaults[model][p].copy()
                r.update(data.pop(p))
                data.update(r)
            graph.add_node(n, data)
        for u,v,d in self.graph.edges_iter():
            data = d.copy()
            graph.add_edge(u, v, **data)
        nx.write_gexf(graph, filename)

    def read_gexf(self, filename):
        self.graph = nx.MultiDiGraph()
        graph = nx.read_gexf(filename)
        for n,d in graph.nodes_iter(data=True):
            model = d.pop('class')
            # neuron and synapse are ambigious at this point
            self.add_neuron(n, model, **d)
        for u,v,d in graph.edges_iter(data=True):
            self.graph.add_edge(u, v, **d)

    @property
    def neuron(self):
        n = {x:d for x,d in self.graph.nodes(True) if self.isneuron(d)}
        return n

    def neurons(self, data=False):
        n = [(x,d) for x,d in self.graph.nodes(True) if self.isneuron(d)]
        if not data:
            n = [x for x,d in n]
        return n

    def isneuron(self, n):
        return issubclass(n['class'], BaseAxonHillockModel.BaseAxonHillockModel) or \
            issubclass(n['class'], BaseMembraneModel.BaseMembraneModel)

    @property
    def synapse(self):
        # TODO: provide pre-/post- neuron hash value
        n = {x:d for x,d in self.graph.nodes(True) if self.issynapse(d)}
        return n

    def synapses(self, data=False):
        # TODO: provide pre-/post- neuron hash value
        n = [(x,d) for x,d in self.graph.nodes(True) if self.issynapse(d)]
        if not data:
            n = [x for x,d in n]
        return n

    def issynapse(self, n):
        return issubclass(n['class'], BaseSynapsekModel.BaseSynapsekModel)
