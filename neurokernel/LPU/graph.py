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
        params = kwargs.pop('params', dict())
        states = kwargs.pop('states', dict())

        if type(model) is str:
            modules = get_all_subclasses(NDComponent.NDComponent)
            modules = {x.__name__: x for x in modules}
            if model in modules:
                model = modules[model]
            else:
                raise TypeError("Unsupported model type %r" % model)

        # assert(issubclass(model, NDComponent))

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

        self.graph.add_node(name,
            {'class':model, 'params':params, 'states':states},
            **attrs)

    def set_model_default(self, model, params, states):
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

    def add_synape(self):
        pass

    def write_gexf(self, filename):
        pass

    def read_gexf(self, filename):
        pass
