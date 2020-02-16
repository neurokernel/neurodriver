"""
Collection of Graph Parser Functions
"""
import networkx as nx
import numbers
import itertools
import copy

def graph_to_dicts(graph: nx.MultiDiGraph,
                   uid_key: str=None,
                   class_key: str='class',
                   remove_edge_id: bool=False):
    """Convert graph of LPU neuron/synapse data to Python data structures.

    :param graph: NetworkX graph containing LPU data.
    :param uid_key: dictionary key that corresponds to uid
    :param class_key: dictionary key that corresponds to class
    :param remove_edge_id: whether to remove id of edges
    :rtype: dict
    :return comp_dict:
        A dictionary of components of which
        keys are model names, and
        values are dictionaries of parameters/attributes associated
        with the model.
        Keys of a dictionary of parameters are the names of them,
        and values of corresponding keys are lists of value of parameters.
        One of the parameters is called 'id' and by default it
        uses the id of the node in the graph.
        If uid_keys is specified, id will use the specified parameter.
        Therefore, comp_dict has the following structure:

        comp_dict = {}
            comp_dict[model_name_1] = {}
                comp_dict[model_name_1][parameter_1] = []
                ...
                comp_dict[model_name_1][parameter_N] = []
                comp_dict[model_name_1][id] = []

            ...

            comp_dict[model_name_M] = {}
                comp_dict[model_name_M][parameter_1] = []
                ...
                comp_dict[model_name_M][parameter_N] = []
                comp_dict[model_name_M][id] = []
    :rtype list:
    :return conns : A list of edges contained in graph describing 
        the relation between components
    """

    comp_dict = {}
    comps = graph.nodes.items()

    all_component_types = list(set([comp[class_key] for uid, comp in comps]))

    for model in all_component_types:
        sub_comps = [comp for comp in comps \
                               if comp[1][class_key] == model]

        all_keys = [set(comp[1]) for comp in sub_comps]
        key_intersection = set.intersection(*all_keys)
        key_union = set.union(*all_keys)

        # For visually checking if any essential parameter is dropped
        ignored_keys = list(key_union-key_intersection)
        if ignored_keys:
            print('parameters of model {} ignored: {}'.format(model, ignored_keys))

        del all_keys

        sub_comp_keys = list(key_intersection)

        if model == 'Port':
            assert('selector' in sub_comp_keys)

        comp_dict[model] = {
            k: [comp[k] for uid, comp in sub_comps] \
            for k in sub_comp_keys if not k in [uid_key, class_key]}

        comp_dict[model]['id'] = [comp[uid_key] if uid_key else uid \
                                  for uid, comp in sub_comps]

        print('Number of {}: {}'.format(model, len(comp_dict[model]['id'])))

    # Extract connections
    conns = graph.edges(data=True)
    if remove_edge_id:
        for pre, post, conn in conns:
            conn.pop('id', None)
    return comp_dict, conns


def lpu_parser(filename):
    """GEXF LPU specification parser.

    Extract LPU specification data from a GEXF file and store it in
    Python data structures.

    :param filename: GEXF filename
    :return: LPU.graph_to_dicts
    """
    graph = nx.read_gexf(filename)
    return graph_to_dicts(graph, remove_edge_id=True)

def extract_in_gpot(comp_dict, uid_key):
    """
    Return selectors of non-spiking input ports.
    """
    if not 'Port' in comp_dict:
        return ('', [])
    a = list(zip(*[(sel,uid) for sel,ptype,io,uid in \
                     zip(comp_dict['Port']['selector'],
                         comp_dict['Port']['port_type'],
                         comp_dict['Port']['port_io'],
                         comp_dict['Port'][uid_key]) if ptype=='gpot' \
              and io=='in']))
    if not a:
        a = ('', [])
    return a


def extract_in_spk(comp_dict, uid_key):
    """
    Return selectors of spiking input ports.
    """
    if not 'Port' in comp_dict: return ('',[])
    a = list(zip(*[(sel,uid) for sel,ptype,io,uid in \
                     zip(comp_dict['Port']['selector'],
                         comp_dict['Port']['port_type'],
                         comp_dict['Port']['port_io'],
                         comp_dict['Port'][uid_key]) if ptype=='spike' \
                     and io=='in']))
    if not a: a = ('',[])
    return a

def extract_out_gpot(comp_dict, uid_key):
    """
    Return selectors of non-spiking output neurons.
    """
    if not 'Port' in comp_dict: return ('',[])
    a = list(zip(*[(sel,uid) for sel,ptype,io,uid in \
                     zip(comp_dict['Port']['selector'],
                         comp_dict['Port']['port_type'],
                         comp_dict['Port']['port_io'],
                         comp_dict['Port'][uid_key]) if ptype=='gpot' \
              and io=='out']))
    if not a: a = ('',[])
    return a

def extract_out_spk(comp_dict, uid_key):
    """
    Return selectors of spiking output neurons.
    """
    if not 'Port' in comp_dict: return ('',[])
    a = list(zip(*[(sel,uid) for sel,ptype,io,uid in \
                     zip(comp_dict['Port']['selector'],
                         comp_dict['Port']['port_type'],
                         comp_dict['Port']['port_io'],
                         comp_dict['Port'][uid_key]) if ptype=='spike' \
              and io=='out']))
    if not a: a = ('',[])
    return a

def extract_sel_in_gpot(comp_dict):
    """
    Return selectors of non-spiking input ports.
    """
    if not 'Port' in comp_dict: return ''
    return ','.join([sel  for sel,ptype,io in \
                     zip(comp_dict['Port']['selector'],
                         comp_dict['Port']['port_type'],
                         comp_dict['Port']['port_io']) \
                         if ptype=='gpot' and io=='in'])


def extract_sel_in_spk(comp_dict):
    """
    Return selectors of spiking input ports.
    """
    if not 'Port' in comp_dict: return ''
    return ','.join([sel  for sel,ptype,io in \
                     zip(comp_dict['Port']['selector'],
                         comp_dict['Port']['port_type'],
                         comp_dict['Port']['port_io']) \
                         if ptype=='spike' and io=='in'])

def extract_sel_out_gpot(comp_dict):
    """
    Return selectors of non-spiking output neurons.
    """
    if not 'Port' in comp_dict: return ''
    return ','.join([sel  for sel,ptype,io in \
                     zip(comp_dict['Port']['selector'],
                         comp_dict['Port']['port_type'],
                         comp_dict['Port']['port_io']) \
                         if ptype=='gpot' and io=='out'])


def extract_sel_out_spk(comp_dict):
    """
    Return selectors of spiking output neurons.
    """
    if not 'Port' in comp_dict: return ''
    return ','.join([sel for sel,ptype,io,uid in \
              zip(comp_dict['Port']['selector'],
                  comp_dict['Port']['port_type'],
                  comp_dict['Port']['port_io']) \
              if ptype=='spike' and io=='out'])


def extract_sel_in(comp_dict):
    """
    Return selectors of all input ports.
    """
    if not 'Port' in comp_dict: return ''
    return ','.join([sel for sel, io in \
                     zip(comp_dict['Port']['selector'],
                         comp_dict['Port']['port_io']) if io=='in'])


def extract_sel_out(comp_dict):
    """
    Return selectors of all output neurons.
    """

    if not 'Port' in comp_dict: return ''
    return ','.join([sel for sel, io in \
                     zip(comp_dict['Port']['selector'],
                         comp_dict['Port']['port_io']) if io=='out'])


def extract_sel_all(comp_dict):
    """
    Return selectors for all input ports and output neurons.

    FIXME!!: `cls.extract_in`, `cls.extract_out` not defined
    """
    return ','.join(filter(None, \
                [cls.extract_in(comp_dict), cls.extract_out(comp_dict)]))

def conv_legacy_graph(g):
    """
    Converts a gexf from legacy neurodriver format to one currently
    supported
    """
    # Find maximum ID in given graph so that we can use it to create new nodes
    # with IDs that don't overlap with those that already exist:
    max_id = 0
    for id in g.nodes():
        if isinstance(id, str):
            if id.isdigit():
                max_id = max(max_id, int(id))
            else:
                raise ValueError('node id must be an integer')
        elif isinstance(id, numbers.Integral):
            max_id = max(max_id, id)
        else:
            raise ValueError('node id must be an integer')
        gen_new_id = next(itertools.count(max_id+1))

    # Create LPU and interface nodes and connect the latter to the former via an
    # Owns edge:
    g_new = nx.MultiDiGraph()

    # Transformation:
    # 1. nonpublic neuron node -> neuron node
    # 2. public neuron node -> neuron node with
    #    output edge to output port
    # 3. input port -> input port
    # 4. synapse edge -> synapse node + 2 edges connecting
    #    transformed original input/output nodes
    edges_to_out_ports = [] # edges to new output port nodes:
    for id, data in g.nodes(data=True):

        # Don't clobber the original graph's data:
        data = copy.deepcopy(data)

        if 'public' in data and data['public']:
            new_id = gen_new_id()
            port_data = {'selector': data['selector'],
                         'port_type': 'spike' if data['spiking'] else 'gpot',
                         'port_io': 'out',
                         'class': 'Port'}
            g_new.add_node(new_id, port_data)
            edges_to_out_ports.append((id, new_id))
            del data['selector']

        if 'model' in data:
            if data['model'] == 'port_in_gpot':
                for a in data:
                    if a!='selector': del data[a]
                data['class'] = 'Port'
                data['port_type'] = 'gpot'
                data['port_io'] = 'in'
            elif data['model'] == 'port_in_spk':
                for a in data:
                    if a!='selector': del data[a]
                data['class'] = 'Port'
                data['port_type'] = 'spike'
                data['port_io'] = 'in'
            else:
                data['class'] = data['model']

            # Don't need to several attributes that are implicit:
            for a in ['model', 'public', 'spiking','extern']:
                if a in data: del data[a]

            g_new.add_node(id, attr_dict=data)

    # Create synapse nodes for each edge in original graph and connect them to
    # the source/dest neuron/port nodes:
    for from_id, to_id, data in g.edges(data=True):
        data = copy.deepcopy(data)
        if data['model'] == 'power_gpot_gpot':
            data['class'] = 'PowerGPotGPot'
        else:
            data['class'] = data['model']
        del data['model']

        if 'id' in data: del data['id']

        new_id = gen_new_id()
        g_new.add_node(new_id, attr_dict=data)
        g_new.add_edge(from_id, new_id, attr_dict={})
        g_new.add_edge(new_id, to_id, attr_dict={})

    # Connect output ports to the neurons that emit data through them:
    for from_id, to_id in edges_to_out_ports:
        g_new.add_edge(from_id, to_id, attr_dict={})

    return g_new


def lpu_parser_legacy(filename):
    """
    TODO: Update
    """
    graph = nx.read_gexf(filename)
    return graph_to_dicts(conv_legacy_graph(graph))
