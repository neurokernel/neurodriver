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
        if type(model) is unicode or type(model) is str:
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

    def connect_port(self, x, y):
        """Connect a port to a node.

        Parameters
        ----------
        x : hashable Python object
            A hashable Python object except None. 'x' has to be an existing
            node or port.
        y : hashable Python object
            A hashable Python object except None. 'y' has to be an existing
            node or port.

        Examples
        --------
        >>> G = Graph()
        >>> G.add_neuron('1', 'LeakyIAF')
        >>> G.add_port('1_port', '/lpu/out/spike/1', port='so')
        >>> G.connect_port('1', '1_port')

        Notes
        -----
        One and only one of 'x' and 'y' must be an existing port, and the
        other has to be an existing node. The direction of the connection will
        be inferred from the port's 'port_io' attribute.
        """
        x_is_port = self.graph.node[x]['class'] == u'Port'
        y_is_port = self.graph.node[y]['class'] == u'Port'

        assert(not(x_is_port == y_is_port))

        if x_is_port:
            port = x
            node = y
        else:
            port = y
            node = x

        if self.graph.node[port]['port_io'] == 'out':
            self.graph.add_edge(node, port)
        else:
            self.graph.add_edge(port, node)

    def add_port(self, name, **kwargs):
        """Add a single port.

        Parameters
        ----------
        name : hashable Python object
            A hashable Python object except None. If 'name' is an existing
            node, a neuron or a synapse, 'name' will be inferred using the
            convention 'name' + "_port".
        selector : string
            A xpath-like string.
        port_type : string
            Either 'spike'/'s' or 'gpot'/'g'.
        port_io : string
            Either 'in'/'i' or 'out'/'o'.
        port : string
            Short notation for 'port_type' and 'port_io'. 'port' can be any
            combination of ['s','g'] cross ['i','o']. For example, 'so' or 'ig'.
            'port' has higher priority than 'port_io' and 'port_type'.
        source_or_target : hashable Python object or None
            The source or the target of the port, depending on 'port_type'.
            If 'name' is an existing node, 'source_or_target' should be given
            as None, and will be replaced by 'name'. If 'source_or_target' is
            not None or inferred from 'name', the connection between the port
            and its correcsponding node will be set up.

        Examples
        --------
        >>> G = Graph()
        >>> G.add_neuron('1', 'LeakyIAF')
        >>> G.add_port('1_port', '/lpu/out/spike/1', port='so')
        >>> G.connect_port('1', '1_port')

        It is recommended to pass an exisitng node as 'name' to add port. By
        doing so, the id of the port will be inferred, and the link between
        the port and the exsiting node will be connected automatically.

        >>> G = Graph()
        >>> G.add_neuron('1', 'LeakyIAF')
        >>> G.add_port('1', port='so')

        Alternatively, one can specify the port name. In this case, it is of
        best practice to provide 'source_or_target'. Otherwise, 'connect_port'
        has to be called later to connect the port and its correcsponding node,
        as demonstrated in the first example.

        >>> G = Graph()
        >>> G.add_neuron('1', 'LeakyIAF')
        >>> G.add_port('special_port', port='so', source_or_target='1')


        Notes
        -----
        A hashable object is one that can be used as a key in a Python
        dictionary. This includes strings, numbers, tuples of strings
        and numbers, etc.
        """
        selector = kwargs.pop('selector', None)
        assert(selector is not None)

        port_io = kwargs.pop('port_io', '')
        port_type = kwargs.pop('port_type', '')
        port = kwargs.pop('port', '')
        source_or_target = kwargs.pop('source_or_target', None)

        is_s, is_g, is_i, is_o = map(lambda x: x in port, ('s','g','i','o'))
        assert(not(is_s and is_g))
        assert(not(is_o and is_i))

        if is_s or port_type == 's':
            port_type = 'spike'
        elif is_g or port_type == 'g':
            port_type = 'gpot'
        assert(port_type == 'gpot' or port_type == 'spike')

        if is_i or port_io == 'i':
            port_io = 'in'
        elif is_o or port_io == 'o':
            port_io = 'out'
        assert(port_io == 'int' or port_io == 'out')

        if name in self.graph:
            assert(source_or_target is None)
            source_or_target = name
            name = "%s_port" % name

        kwargs['class'] = u'Port'
        self.graph.add_node(name,
            kwargs,
            port_io = port_io,
            port_type = port_type,
            selector = selector)

        if source_or_target is not None:
            assert(source_or_target in self.graph)
            if port_io == 'out':
                self.graph.add_edge(source_or_target, name)
            else:
                self.graph.add_edge(name, source_or_target)

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
            if data['class'] != u'Port':
                model = data.pop('class')
                data['class'] = model.__name__.encode('utf-8')
                for p in ('params', 'states'):
                    r = self.modelDefaults[model][p].copy()
                    r.update(data.pop(p))
                    data.update(r)
            graph.add_node(n, data)
        for u,v,d in self.graph.edges_iter(data=True):
            data = d.copy()
            graph.add_edge(u, v, **data)
        nx.write_gexf(graph, filename)

    def read_gexf(self, filename):
        self.graph = nx.MultiDiGraph()
        graph = nx.read_gexf(filename)
        for n,d in graph.nodes_iter(data=True):
            if d['class'].encode('utf-8') == u'Port':
                self.add_port(n, **d)
            else:
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

if __name__ == "__main__":
    from neurokernel.LPU.graph import Graph

    a = Graph()
    a.add_neuron('1', 'LeakyIAF')
    a.add_neuron('2', 'LeakyIAF')
    a.add_port('1',port='so', selector='xx')
    a.add_synapse('1--2', '1', '2', 'AlphaSynapse')
    a.write_gexf('temp.gexf')

    b = Graph()
    b.read_gexf('temp.gexf')
