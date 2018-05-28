#!/usr/bin/env python

"""
Local Processing Unit (LPU) with plugin support for various neuron/synapse models.
"""
import collections
import numbers
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.elementwise as elementwise

import numpy as np
import networkx as nx

#import time

import copy
import itertools
import numbers

# Work around bug in networkx < 1.9 that causes networkx to choke on GEXF
# files with boolean attributes that contain the strings 'True' or 'False'
# (bug already observed in https://github.com/networkx/networkx/pull/971)
nx.readwrite.gexf.GEXF.convert_bool['false'] = False
nx.readwrite.gexf.GEXF.convert_bool['False'] = False
nx.readwrite.gexf.GEXF.convert_bool['true'] = True
nx.readwrite.gexf.GEXF.convert_bool['True'] = True

from neurokernel.mixins import LoggerMixin
from neurokernel.core_gpu import Module, CTRL_TAG, GPOT_TAG, SPIKE_TAG
from neurokernel.tools.gpu import get_by_inds

from types import *
from collections import Counter

from .utils.simpleio import *
from .utils import parray

from .NDComponents import *
from .MemoryManager import MemoryManager

import pdb

PORT_IN_GPOT = 'port_in_gpot'
PORT_IN_SPK = 'port_in_spk'
PORT_OUT_GPOT = 'port_out_gpot'
PORT_OUT_SPK = 'port_out_spk'

class LPU(Module):
    @staticmethod
    def conv_legacy_graph(g):
        """
        Converts a gexf from legacy neurodriver format to one currently
        supported
        """


        # Find maximum ID in given graph so that we can use it to create new nodes
        # with IDs that don't overlap with those that already exist:
        max_id = 0
        for id in g.nodes():
            if isinstance(id, basestring):
                if id.isdigit():
                    max_id = max(max_id, int(id))
                else:
                    raise ValueError('node id must be an integer')
            elif isinstance(id, numbers.Integral):
                max_id = max(max_id, id)
            else:
                raise ValueError('node id must be an integer')
            gen_new_id = itertools.count(max_id+1).next

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
                    for a in data.keys():
                        if a!='selector': del data[a]
                    data['class'] = 'Port'
                    data['port_type'] = 'gpot'
                    data['port_io'] = 'in'
                elif data['model'] == 'port_in_spk':
                    for a in data.keys():
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

    @staticmethod
    def graph_to_dicts(graph, uid_key=None, class_key='class'):
        """
        Convert graph of LPU neuron/synapse data to Python data structures.

        Parameters
        ----------
        graph : networkx.MultiDiGraph
            NetworkX graph containing LPU data.

        Returns
        -------
        comp_dict : dict
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

        conns : list
            A list of edges contained in graph describing the relation
            between components

        Example
        -------
        TODO: Update

        Notes
        -----
        TODO: Update
        """

        comp_dict = {}
        comps = graph.node.items()

        all_component_types = list(set([comp[class_key] for uid, comp in comps]))

        for model in all_component_types:
            sub_comps = [comp for comp in comps \
                                   if comp[1][class_key] == model]

            all_keys = [set(comp[1].keys()) for comp in sub_comps]
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

#        for id, comp in comps:
#            model = comp[class_key]
#
#            # For port, make sure selector is specified
#            if model == 'Port':
#                assert('selector' in comp.keys())
#
#            # if the neuron model does not appear before, add it into n_dict
#            if model not in comp_dict:
#                comp_dict[model] = {k:[] for k in comp.keys() + ['id']}
#
#            # Same model should have the same attributes
#            if not set(comp_dict[model].keys()) == set(comp.keys() + ['id']):
#                raise KeyError("keys of component does not match with that of "+\
#                               model+": "+ str(set(comp_dict[model].keys())) +
#                               str(set(comp.keys() + ['id'])))
#
#            # add data to the subdictionary of comp_dict
#            for key in comp.iterkeys():
#                if not key==uid_key:
#                    comp_dict[model][key].append( comp[key] )
#            if uid_key:
#                comp_dict[model]['id'].append(comp[uid_key])
#            else:
#                comp_dict[model]['id'].append( id )
#
#        # Remove duplicate model information:
#        for val in comp_dict.itervalues(): val.pop(class_key)

        # Extract connections
        conns = graph.edges(data=True)
        return comp_dict, conns

    @staticmethod
    def lpu_parser(filename):
        """
        GEXF LPU specification parser.

        Extract LPU specification data from a GEXF file and store it in
        Python data structures.
        TODO: Update

        Parameters
        ----------
        filename : str
            GEXF filename.

        Returns
        -------
        TODO: Update
        """

        graph = nx.read_gexf(filename)
        return LPU.graph_to_dicts(graph)

    @staticmethod
    def lpu_parser_legacy(filename):
        """
        TODO: Update
        """

        graph = nx.read_gexf(filename)
        return LPU.graph_to_dicts(LPU.conv_legacy_graph(graph))

    @classmethod
    def extract_in_gpot(cls, comp_dict, uid_key):
        """
        Return selectors of non-spiking input ports.
        """
        if not 'Port' in comp_dict: return ('',[])
        a = zip(*[(sel,uid) for sel,ptype,io,uid in \
                         zip(comp_dict['Port']['selector'],
                             comp_dict['Port']['port_type'],
                             comp_dict['Port']['port_io'],
                             comp_dict['Port'][uid_key]) if ptype=='gpot' \
                  and io=='in'])
        if not a: a = ('',[])
        return a

    @classmethod
    def extract_in_spk(cls, comp_dict, uid_key):
        """
        Return selectors of spiking input ports.
        """
        if not 'Port' in comp_dict: return ('',[])
        a = zip(*[(sel,uid) for sel,ptype,io,uid in \
                         zip(comp_dict['Port']['selector'],
                             comp_dict['Port']['port_type'],
                             comp_dict['Port']['port_io'],
                             comp_dict['Port'][uid_key]) if ptype=='spike' \
                         and io=='in'])
        if not a: a = ('',[])
        return a

    @classmethod
    def extract_out_gpot(cls, comp_dict, uid_key):
        """
        Return selectors of non-spiking output neurons.
        """
        if not 'Port' in comp_dict: return ('',[])
        a = zip(*[(sel,uid) for sel,ptype,io,uid in \
                         zip(comp_dict['Port']['selector'],
                             comp_dict['Port']['port_type'],
                             comp_dict['Port']['port_io'],
                             comp_dict['Port'][uid_key]) if ptype=='gpot' \
                  and io=='out'])
        if not a: a = ('',[])
        return a

    @classmethod
    def extract_out_spk(cls, comp_dict, uid_key):
        """
        Return selectors of spiking output neurons.
        """
        if not 'Port' in comp_dict: return ('',[])
        a = zip(*[(sel,uid) for sel,ptype,io,uid in \
                         zip(comp_dict['Port']['selector'],
                             comp_dict['Port']['port_type'],
                             comp_dict['Port']['port_io'],
                             comp_dict['Port'][uid_key]) if ptype=='spike' \
                  and io=='out'])
        if not a: a = ('',[])
        return a

    @classmethod
    def extract_sel_in_gpot(cls, comp_dict):
        """
        Return selectors of non-spiking input ports.
        """
        if not 'Port' in comp_dict: return ''
        return ','.join([sel  for sel,ptype,io in \
                         zip(comp_dict['Port']['selector'],
                             comp_dict['Port']['port_type'],
                             comp_dict['Port']['port_io']) \
                             if ptype=='gpot' and io=='in'])

    @classmethod
    def extract_sel_in_spk(cls, comp_dict):
        """
        Return selectors of spiking input ports.
        """
        if not 'Port' in comp_dict: return ''
        return ','.join([sel  for sel,ptype,io in \
                         zip(comp_dict['Port']['selector'],
                             comp_dict['Port']['port_type'],
                             comp_dict['Port']['port_io']) \
                             if ptype=='spike' and io=='in'])

    @classmethod
    def extract_sel_out_gpot(cls, comp_dict):
        """
        Return selectors of non-spiking output neurons.
        """
        if not 'Port' in comp_dict: return ''
        return ','.join([sel  for sel,ptype,io in \
                         zip(comp_dict['Port']['selector'],
                             comp_dict['Port']['port_type'],
                             comp_dict['Port']['port_io']) \
                             if ptype=='gpot' and io=='out'])

    @classmethod
    def extract_sel_out_spk(cls, comp_dict):
        """
        Return selectors of spiking output neurons.
        """
        if not 'Port' in comp_dict: return ''
        return ','.join([sel for sel,ptype,io,uid in \
                  zip(comp_dict['Port']['selector'],
                      comp_dict['Port']['port_type'],
                      comp_dict['Port']['port_io']) \
                  if ptype=='spike' and io=='out'])

    @classmethod
    def extract_sel_in(cls, comp_dict):
        """
        Return selectors of all input ports.
        """
        if not 'Port' in comp_dict: return ''
        return ','.join([sel for sel, io in \
                         zip(comp_dict['Port']['selector'],
                             comp_dict['Port']['port_io']) if io=='in'])

    @classmethod
    def extract_sel_out(cls, comp_dict):
        """
        Return selectors of all output neurons.
        """

        if not 'Port' in comp_dict: return ''
        return ','.join([sel for sel, io in \
                         zip(comp_dict['Port']['selector'],
                             comp_dict['Port']['port_io']) if io=='out'])

    @classmethod
    def extract_sel_all(cls, comp_dict):
        """
        Return selectors for all input ports and output neurons.
        """

        return ','.join(filter(None, \
                    [cls.extract_in(comp_dict), cls.extract_out(comp_dict)]))


    def __init__(self, dt, comp_dict, conn_list, device=0, input_processors=[],
                 output_processors=[], ctrl_tag=CTRL_TAG, gpot_tag=GPOT_TAG,
                 spike_tag=SPIKE_TAG, rank_to_id=None, routing_table=None,
                 uid_key='id', debug=False, columns=['io', 'type', 'interface'],
                 cuda_verbose=False, time_sync=False, default_dtype=np.double,
                 control_inteface=None, id=None, extra_comps=[]):

        LoggerMixin.__init__(self, 'LPU {}'.format(id))

        assert('io' in columns)
        assert('type' in columns)
        assert('interface' in columns)
        self.LPU_id = id
        self.dt = dt
        self.time = 0
        self.debug = debug
        self.device = device
        self.default_dtype = default_dtype
        self.control_inteface = control_inteface
        if cuda_verbose:
            self.compile_options = ['--ptxas-options=-v']
        else:
            self.compile_options = []

        if not isinstance(input_processors, list):
            input_processors = [input_processors]
        if not isinstance(output_processors, list):
            input_processors = [output_processors]

        self.output_processors = output_processors
        self.input_processors = input_processors

        self.gen_uids = []
        self.uid_key = uid_key

        # Load all NDComponents:
        self._load_components(extra_comps=extra_comps)

        # Ignore models without implementation
        models_to_be_deleted = []
        for model in comp_dict:
            if not model in self._comps and not model in ['Port','Input']:
                self.log_info("Ignoring Model %s: Can not find implementation"
                              % model)
                models_to_be_deleted.append(model)
        for model in models_to_be_deleted:
            del comp_dict[model]

        # Assume zero delay by default
        self.variable_delay_map = {}

        # Generate a uid to model map of components
        self.uid_model_map = {}
        for model,attribs in comp_dict.iteritems():
            for i,uid in enumerate(attribs[uid_key]):
                self.uid_model_map[uid] = model


        # Map from post synaptic component to aggregator uid
        agg_map = {}
        agg = {}

        #start = time.time()

        conns = []
        self.in_port_vars = {}
        self.out_port_conns = []
        comp_uid_order = {}
        for model, attribs in comp_dict.items():
            comp_uid_order[model] = {uid: i for i, uid in enumerate(attribs[uid_key])}
        # Process connections between components, remove inconsitent connections
        # calculate required delays, infer variable if required
        for conn in conn_list:
            pre = conn[0]
            post = conn[1]
            if not (pre in self.uid_model_map and post in self.uid_model_map): continue
            pre_model = self.uid_model_map[pre]
            post_model = self.uid_model_map[post]

            pre_updates = self._comps[pre_model]['updates'] \
                          if not pre_model=='Port' else []
            post_accesses = self._comps[post_model]['accesses'] \
                          if not post_model=='Port' else []

            data = conn[2] if len(conn)>2 else {}

            # Update delay
            delay = max(int(round((data['delay']/dt))) \
                    if 'delay' in data else 0, 1) - 1
            data['delay'] = delay

            if post_model == 'Aggregator':
                agg_map[post] = post
                reverse_key = None
                s = (set(['reverse','Vr','VR','reverse_potential'])&
                     set(data.keys()))
                if s: reverse_key = s.pop()
                if reverse_key:
                    reverse = data[reverse_key]
                else:
                    # else look in the attibutes of the synapse
                    s = (set(['reverse','Vr','VR','reverse_potential'])&
                         set(comp_dict[pre_model].keys()))
                    if s: reverse_key = s.pop()
                    if reverse_key:
                        reverse = comp_dict[pre_model][reverse_key][\
                                comp_uid_order[pre_model][pre]]
                        if 'g' in pre_updates:
                            data['reverse'] = reverse
                    else:
                        if 'g' in pre_updates:
                            self.log_info('Assuming reverse potential ' +
                                          'to be zero for connection from' +
                                          '%s to %s'%(pre,post))
                            data['reverse'] = 0
                        reverse = 0

                if post in agg:
                    if 'g' in pre_updates:
                        agg[post].append({'pre':pre,'reverse':reverse,
                                          'variable':'g'})
                    elif 'V' in pre_updates:
                        agg[post].append({'pre':pre, 'variable':'V'})
                else:
                    # Make sure aggregator has access to postsynaptic voltage
                    if 'g' in pre_updates:
                        agg[post] = [{'pre':pre,'reverse':reverse,'variable':'g'}]
                    elif 'V' in pre_updates:
                        agg[post] = [{'pre':pre,'variable':'V'}]

                if 'g' in pre_updates:
                    agg[post][-1].update({k:v for k,v in data.items()})
                    self.variable_delay_map['g'] = max(data['delay'],
                                    self.variable_delay_map['g'] if 'g' in \
                                    self.variable_delay_map else 0)


            # Ensure consistency
            # Insert Aggregator between g->V if required. Assume 'reverse' or
            # 'Vr' or 'VR' or 'reverse_potential' id present as a param in the
            # synapse in that case
            if not (set(pre_updates)&set(post_accesses)):
                if 'g' in pre_updates and 'I' in post_accesses:
                    # First look for reverse in the attributes of the edge
                    reverse_key = None
                    s = (set(['reverse','Vr','VR','reverse_potential'])&
                         set(data.keys()))
                    if s: reverse_key = s.pop()
                    if reverse_key:
                        reverse = data[reverse_key]
                    else:
                        # else look in the attibutes of the synapse
                        s = (set(['reverse','Vr','VR','reverse_potential'])&
                             set(comp_dict[pre_model].keys()))
                        if s: reverse_key = s.pop()
                        if reverse_key:
                            reverse = comp_dict[pre_model][reverse_key][\
                                    comp_uid_order[pre_model][pre]]
                        else:
                            self.log_info('Assuming reverse potential ' +
                                          'to be zero for connection from' +
                                          '%s to %s'%(pre,post))
                            reverse = 0
                    if post in agg:
                        agg[post].append({'pre':pre,'reverse':reverse,
                                          'variable':'g'})
                    else:
                        # Make sure aggregator has access to postsynaptic voltage
                        agg[post] = [{'pre':post,'variable':'V'},
                                     {'pre':pre,'reverse':reverse,'variable':'g'}]
                    agg[post][-1].update({k:v for k,v in data.items()})
                    if post not in agg_map:
                        uid = self.generate_uid()
                        agg_map[post] = uid
                        self.gen_uids.append(uid)
                    self.variable_delay_map['g'] = max(data['delay'],
                                    self.variable_delay_map['g'] if 'g' in \
                                    self.variable_delay_map else 0)

                elif pre_model == 'Port':
                    if not 'variable' in data:
                        data['variable'] = post_accesses[0]
                    if not data['variable'] in self.in_port_vars:
                        self.in_port_vars[data['variable']] = []
                    if pre not in self.in_port_vars[data['variable']]:
                        self.in_port_vars[data['variable']].append(pre)
                    conns.append((pre, post, data))
                    self.variable_delay_map[data['variable']] = max(data['delay'],
                            self.variable_delay_map[data['variable']] if \
                            data['variable'] in self.variable_delay_map else 0)
                elif post_model == 'Port':
                    if not 'variable' in data:
                        data['variable'] = pre_updates[0]
                    self.out_port_conns.append((pre, post, data['variable']))
                else:
                    self.log_info("Ignoring connection %s -> %s"%(pre,post))
                continue

            var = data['variable'] if 'variable' in data else None
            if not var:
                var = (set(pre_updates)&set(post_accesses)).pop()
            elif not (var in pre_updates and var in post_accesses):
                continue
            data['variable'] = var
            self.variable_delay_map[data['variable']] = max(data['delay'],
                            self.variable_delay_map[data['variable']] if \
                            data['variable'] in self.variable_delay_map else 0)
            # connection to Aggregator will be added later
            if not post_model == 'Aggregator':
                conns.append((pre,post,data))

        #print self.LPU_id, "step 1:", time.time()-start

        if agg and not 'Aggregator' in comp_dict:
            comp_dict['Aggregator'] = {uid_key: []}

        # Add updated aggregator components to component dictionary
        # and create connections for aggregator
        for post, conn_list  in agg.items():
            uid = agg_map[post]
            if uid not in comp_dict['Aggregator'][uid_key]:
                keys = [k for k in comp_dict['Aggregator'].keys() if k != uid_key]
                comp_dict['Aggregator'][uid_key].append(uid)
                self.uid_model_map[uid] = 'Aggregator'
                for k in keys:
                    comp_dict['Aggregator'][k].append(str(uid))
            for conn in conn_list:
                conns.append((conn['pre'],uid,{k:v for k,v in conn.items()
                                               if k!='pre'}))
            # Add a 'I' connection between Aggregator and neuron if they are
            # automatically generated.
            # This can be checking if the 'pre' attribute in the item
            # in conn_list with 'variable' 'V' is the same neuron as post
            if post == [tmp['pre'] for tmp in conn_list if tmp['variable']=='V'][0]:
                conns.append((uid,post,{'variable':'I', 'delay': 0}))

        #print self.LPU_id, "step 2:", time.time()-start

        self.conn_dict = {}

        # RePackage connections
        for (pre, post, data) in conns:
            if not post in self.conn_dict:
                self.conn_dict[post] = {}
            var = data['variable']
            data.pop('variable')
            if not var in self.conn_dict[post]:
                self.conn_dict[post][var] = {k:[] for k in ['pre'] + data.keys()}
            self.conn_dict[post][var]['pre'].append(pre)
            for k in data: self.conn_dict[post][var][k].append(data[k])

        #print self.LPU_id, "step 3:", time.time()-start

        # Add connections for component with no incoming connections
        for uid, model in self.uid_model_map.iteritems():
            if not model == 'Port':
                var = self._comps[model]['accesses'][0]
            else:
                var = ''
            if ((not uid in self.conn_dict or not var in self.conn_dict[uid])
                and not model == 'Port'):
                pre = self.generate_uid(input=True)
                self.gen_uids.append(pre)
                if not var in self.variable_delay_map:
                    self.variable_delay_map[var]=0
                if not uid in self.conn_dict: self.conn_dict[uid] = {}
                if model == 'Aggregator' and var == 'g':
                    self.conn_dict[uid][var] = {'pre':[pre],'delay':[0],
                                                'reverse': [0]} #'id': [0],
                else:
                    self.conn_dict[uid][var] = {'pre':[pre],'delay':[0]}
                if not 'Input' in comp_dict:
                    comp_dict['Input'] = {}
                if not var in comp_dict['Input']:
                    comp_dict['Input'][var] = {self.uid_key: []}
                comp_dict['Input'][var][self.uid_key].append(pre)

        #print self.LPU_id, "step 4:", time.time()-start

        # Optimize ordering (TODO)
        self.uid_ind_map = {m:{uid:i for i,uid in enumerate(n[uid_key])}
                            for m,n in comp_dict.items() if not m=='Input'}

        if 'Input' in comp_dict:
            self.uid_ind_map['Input'] = {var:{uid:i for i, uid in enumerate(d[uid_key])}
                                         for var, d in comp_dict['Input'].items()}

        # Reorder components
        for m, n in comp_dict.items():
            if m=='Input':
                for var, d in n.items():
                    order = np.argsort([self.uid_ind_map[m][var][uid] for uid in d[uid_key]])
                    d[uid_key] = [d[uid_key][i] for i in order]
                continue

            order = np.argsort([self.uid_ind_map[m][uid] for uid in n[uid_key]])
            for k in n.keys():
                n[k] = [n[k][i] for i in order]

        # Reorder input port variables
        for var, uids in self.in_port_vars.items():
            order = np.argsort([self.uid_ind_map['Port'][uid] for uid in uids])
            self.in_port_vars[var] = [uids[i] for i in order]

        #print self.LPU_id, "step 5:", time.time()-start

        # Try to figure out order of stepping through components
        # If a loop of dependencies is present, update order behaviour is undefined
        models = comp_dict.keys()
        try:
            models.remove('Port')
        except:
            pass

        try:
            models.remove('Input')
        except:
            pass


        deps = {i:[] for i in range(len(models))}
        for i in range(len(models)):
            for j in range(i+1,len(models)):
                in12 = set(self._comps[models[i]]['updates'])&\
                       set(self._comps[models[j]]['accesses'])
                in21 = set(self._comps[models[i]]['accesses'])&\
                       set(self._comps[models[j]]['updates'])
                if in12 or in21:
                    if len(in12) > len(in21):
                        deps[j].append(i)
                    else:
                        deps[i].append(j)


        self.exec_order = []
        for i, model in enumerate(models):
            if not model in self.exec_order: self.exec_order.append(model)
            for j in deps[i]:
                try:
                    if self.exec_order.index(models[j]) > \
                       self.exec_order.index(model):
                        self.exec_order.remove(models[j])
                        self.exec_order.insert(self.exec_order.index(model),
                                               models[j])
                except ValueError:
                    self.exec_order.insert(self.exec_order.index(model),
                                           models[j])

        var_mod = {}
        for i, model in enumerate(models):
            for var in self._comps[model]['updates']:
                if not var in var_mod: var_mod[var] = []
                var_mod[var].append(model)

        self.model_var_inj = {}
        for var, models in var_mod.items():
            i = 0
            for model in models:
                i = max(self.exec_order.index(model),i)
            if not self.exec_order[i] in self.model_var_inj:
                self.model_var_inj[self.exec_order[i]] = []
            self.model_var_inj[self.exec_order[i]].append(var)

        #Variables not updated by any component (for example those coming from
        #external input or Ports) are slated to be injected at the end of a step
        for var in self.variable_delay_map:
            if not var in var_mod:
                if not self.exec_order[-1] in self.model_var_inj:
                    self.model_var_inj[self.exec_order[-1]] = []
                self.model_var_inj[self.exec_order[-1]].append(var)

        # Get selectors of input ports:
        self.sel_in_gpot, self.in_gpot_uids = self.extract_in_gpot(comp_dict,
                                                                   self.uid_key)
        self.sel_in_spk, self.in_spk_uids = self.extract_in_spk(comp_dict,
                                                                self.uid_key)

        sel_in = ','.join(filter(None, [','.join(self.sel_in_gpot),
                                        ','.join(self.sel_in_spk)]))

        # Get selectors of output neurons:
        self.sel_out_gpot, self.out_gpot_uids = self.extract_out_gpot(comp_dict,
                                                                      self.uid_key)
        self.sel_out_spk, self.out_spk_uids = self.extract_out_spk(comp_dict,
                                                                   self.uid_key)

        sel_out = ','.join(filter(None, [','.join(self.sel_out_gpot),
                                         ','.join(self.sel_out_spk)]))
        sel_gpot = ','.join(filter(None, [','.join(self.sel_in_gpot),
                                          ','.join(self.sel_out_gpot)]))
        sel_spk = ','.join(filter(None, [','.join(self.sel_in_spk),
                                         ','.join(self.sel_out_spk)]))
        sel = ','.join(filter(None, [sel_gpot, sel_spk]))


        # Save component parameters data in the form
        # [('Model0', {'attrib0': [..], 'attrib1': [..]}), ('Model1', ...)]
        self.comp_list = comp_dict.items()
        self.models = {m:i for i,(m,_) in enumerate(self.comp_list)}

        # Number of components of each model:
        self.model_num = [len(n[uid_key]) if not m=='Input' else
                          len(sum([d[uid_key] for d in n.values()],[]))
                          for m, n in self.comp_list]

        data_gpot = np.zeros(len(self.in_gpot_uids)+len(self.out_gpot_uids),
                             self.default_dtype)
        data_spike = np.zeros(len(self.in_spk_uids)+len(self.out_spk_uids)
                              ,np.int32)
        super(LPU, self).__init__(sel=sel, sel_in=sel_in, sel_out=sel_out,
                                  sel_gpot=sel_gpot, sel_spike=sel_spk,
                                  data_gpot=data_gpot, data_spike=data_spike,
                                  columns=columns, ctrl_tag=ctrl_tag, gpot_tag=gpot_tag,
                                  spike_tag=spike_tag, id=self.LPU_id,
                                  rank_to_id=rank_to_id, routing_table=routing_table,
                                  device=device, debug=debug, time_sync=time_sync)



        # Integer indices in port map data arrays corresponding to input/output
        # gpot/spiking ports:
        self.in_gpot_inds = np.array(self.pm['gpot'].ports_to_inds(\
                                    ','.join(self.sel_in_gpot)), dtype=np.int32)
        self.out_gpot_inds = np.array(self.pm['gpot'].ports_to_inds(\
                                    ','.join(self.sel_out_gpot)), dtype=np.int32)
        self.in_spk_inds = np.array(self.pm['spike'].ports_to_inds(\
                                    ','.join(self.sel_in_spk)), dtype=np.int32)
        self.out_spk_inds = np.array(self.pm['spike'].ports_to_inds(\
                                    ','.join(self.sel_out_spk)), dtype=np.int32)

    def generate_uid(self, input=False):
        if input:
            uid = 'input_' + str(np.random.randint(100000))
        else:
            uid = 'auto_' + str(np.random.randint(100000))
        while uid in self.gen_uids:
            if input:
                uid = 'input_' + str(np.random.randint(100000))
            else:
                uid = 'auto_' + str(np.random.randint(100000))
        return uid

    def pre_run(self):
        #start = time.time()
        super(LPU, self).pre_run()
        #print self.LPU_id, "step 6:", time.time()-start
        self.memory_manager = MemoryManager()
        self.init_variable_memory()
        #print self.LPU_id, "step 7:", time.time()-start
        self.process_connections()
        #print self.LPU_id, "step 8:", time.time()-start
        self.init_parameters()
        #print self.LPU_id, "step 9:", time.time()-start

        self.components = {}
        # Instantiate components
        for model in self.models:
            if model in ['Port','Input']: continue
            self.components[model] = self._instantiate_component(model)
            update_pointers = {}
            for var in self._comps[model]['updates']:
                buff = self.memory_manager.get_buffer(var)
                mind = self.memory_manager.variables[var]['models'].index(model)
                shift = self.memory_manager.variables[var]['cumlen'][mind]
                update_pointers[var] = int(buff.gpudata)+(buff.current*buff.ld+\
                                            shift)*buff.dtype.itemsize
            self.components[model].pre_run(update_pointers)
            for var in self._comps[model]['updates']:
                buff = self.memory_manager.get_buffer(var)
                mind = self.memory_manager.variables[var]['models'].index(model)
                shift = self.memory_manager.variables[var]['cumlen'][mind]
                for j in range(buff.buffer_length):
                    if j is not buff.current:
                        cuda.memcpy_dtod(
                            int(buff.gpudata)+(j*buff.ld+\
                                                shift)*buff.dtype.itemsize,
                            int(buff.gpudata)+(buff.current*buff.ld+\
                                                shift)*buff.dtype.itemsize,
                            buff.dtype.itemsize*self.model_num[self.models[model]])

        #print self.LPU_id, "step 10:", time.time()-start

        # Setup ports
        self._setup_input_ports()
        self._setup_output_ports()

        for p in self.input_processors:
            p.LPU_obj = self
            p._pre_run()

        for p in self.output_processors:
            p.LPU_obj = self
            p._pre_run()

        if self.control_inteface: self.control_inteface.register(self)

    # TODO: optimize the order of self.out_port_conns beforehand
    def _setup_output_ports(self):
        self.out_port_inds_gpot = {}
        self.out_var_inds_gpot = {}
        self.out_port_inds_spk = {}
        self.out_var_inds_spk = {}
        for pre_uid, post_uid, var in self.out_port_conns:
            if not var in self.out_port_inds_gpot:
                self.out_port_inds_gpot[var] = []
                self.out_var_inds_gpot[var] = []
                self.out_port_inds_spk[var] = []
                self.out_var_inds_spk[var] = []
            ind = self.memory_manager.variables[var]['uids'][pre_uid]
            if post_uid in self.out_gpot_uids:
                self.out_port_inds_gpot[var].append(self.out_gpot_inds[\
                                            self.out_gpot_uids.index(post_uid)])
                self.out_var_inds_gpot[var].append(ind)
            else:
                self.out_port_inds_spk[var].append(self.out_spk_inds[\
                                            self.out_spk_uids.index(post_uid)])
                self.out_var_inds_spk[var].append(ind)

        for var in self.out_port_inds_gpot.keys():
            if not self.out_port_inds_gpot[var]:
                del self.out_port_inds_gpot[var]
                del self.out_var_inds_gpot[var]
            else:
                self.out_port_inds_gpot[var] = garray.to_gpu(\
                        np.array(self.out_port_inds_gpot[var],np.int32))
                self.out_var_inds_gpot[var] = garray.to_gpu(\
                        np.array(self.out_var_inds_gpot[var],np.int32))
        for var in self.out_port_inds_spk.keys():
            if not self.out_port_inds_spk[var]:
                del self.out_port_inds_spk[var]
                del self.out_var_inds_spk[var]
            else:
                self.out_port_inds_spk[var] = garray.to_gpu(\
                        np.array(self.out_port_inds_spk[var],np.int32))
                self.out_var_inds_spk[var] = garray.to_gpu(\
                        np.array(self.out_var_inds_spk[var],np.int32))

    def _setup_input_ports(self):
        self.port_inds_gpot = {}
        self.var_inds_gpot = {}
        self.port_inds_spk = {}
        self.var_inds_spk = {}
        for var, uids in self.in_port_vars.items():
            self.port_inds_gpot[var] = []
            self.var_inds_gpot[var] = []
            self.port_inds_spk[var] = []
            self.var_inds_spk[var] = []
            mind = self.memory_manager.variables[var]['models'].index('Port')
            shift = self.memory_manager.variables[var]['cumlen'][mind]
            # The following assumes the intersection of set of variables
            # accessed via spiking with those accessed via gpot ports is null
            for i,uid in enumerate(uids):
                if uid in self.in_gpot_uids:
                    self.port_inds_gpot[var].append(self.in_gpot_inds[\
                                            self.in_gpot_uids.index(uid)])
                    self.var_inds_gpot[var].append(i + shift)
                else:
                    self.port_inds_spk[var].append(self.in_spk_inds[\
                                            self.in_spk_uids.index(uid)])
                    self.var_inds_spk[var].append(i + shift)
        for var in self.port_inds_gpot.keys():
            if not self.port_inds_gpot[var]:
                del self.port_inds_gpot[var]
                del self.var_inds_gpot[var]
            else:
                self.port_inds_gpot[var] = garray.to_gpu(\
                        np.array(self.port_inds_gpot[var],np.int32))
                self.var_inds_gpot[var] = garray.to_gpu(\
                        np.array(self.var_inds_gpot[var],np.int32))
        for var in self.port_inds_spk.keys():
            if not self.port_inds_spk[var]:
                del self.port_inds_spk[var]
                del self.var_inds_spk[var]
            else:
                self.port_inds_spk[var] = garray.to_gpu(\
                        np.array(self.port_inds_spk[var],np.int32))
                self.var_inds_spk[var] = garray.to_gpu(\
                        np.array(self.var_inds_spk[var],np.int32))


    def init_parameters(self):
        for m, n in self.comp_list:
            if not m in ['Port','Input']:
                nn = n.copy()
                nn.pop(self.uid_key)
                # copy integer and boolean parameters into separate dictionary
                nn_int = {k:v for k, v in nn.iteritems() if (isinstance(v, list)
                            and len(v) and type(v[0]) in [int, long, bool])}
                nn_rest = {k:v for k, v in nn.iteritems() if (
                           (not isinstance(v, list)) or (len(v) and
                           type(v[0]) not in [int, long, bool]))}
                if nn_int:
                    self.memory_manager.params_htod(m, nn_int, np.int32)
                if nn_rest:
                    self.memory_manager.params_htod(m, nn_rest,
                                                    self.default_dtype)

    def init_variable_memory(self):
        var_info = {}
        for (model, attribs) in self.comp_list:
            if model in ['Port']: continue
            # Add memory for external inputs if required
            if model == 'Input':
                for var, d in attribs.iteritems():
                    if not var in var_info:
                        var_info[var] = {'models':[],'len':[],'delay':0,'uids':[]}
                    var_info[var]['models'].append('Input')
                    var_info[var]['len'].append(len(d[self.uid_key]))
                    var_info[var]['uids'].extend(d[self.uid_key])
                continue
            for var in self._comps[model]['updates']:
                if not var in var_info:
                    var_info[var] = {'models':[],'len':[],'delay':0,'uids':[]}
                var_info[var]['models'].append(model)
                var_info[var]['len'].append(len(attribs[self.uid_key]))
                var_info[var]['uids'].extend(attribs[self.uid_key])

        # Add memory for input ports
        for var in self.in_port_vars.keys():
            if not var in var_info:
                var_info[var] = {'models':[],'len':[],'delay':0,'uids':[]}
            var_info[var]['models'].append('Port')
            var_info[var]['len'].append(len(self.in_port_vars[var]))
            var_info[var]['uids'].extend(self.in_port_vars[var])



        for var in self.variable_delay_map.keys():
            var_info[var]['delay'] = self.variable_delay_map[var]

        for var, d in var_info.items():
            d['cumlen'] = np.cumsum([0]+d['len'])
            d['uids'] = {uid:i for i, uid in enumerate(d['uids'])}
            self.memory_manager.memory_alloc(var, d['cumlen'][-1], d['delay']+2,\
                dtype=self.default_dtype if not var=='spike_state' else np.int32,
                info=d)

    def process_connections(self):
        for (model, attribs) in self.comp_list:
            if model in ['Port','Input']: continue
            pre = {var:[] for var in self._comps[model]['accesses']}
            npre = {var:[] for var in self._comps[model]['accesses']}
            data = {var:{} for var in self._comps[model]['accesses']}
            for uid in attribs[self.uid_key]:
                cnt = {var:0 for var in self._comps[model]['accesses']}
                if uid in self.conn_dict:
                    for var in self.conn_dict[uid]:
                        for i in range(len(self.conn_dict[uid][var]['pre'])):
                            # Figure out index of the precomponent in the
                            # particular variable memory
                            p = self.conn_dict[uid][var]['pre'][i]
                            ind = self.memory_manager.variables[var]['uids'][p]
                            pre[var].append(ind)

                            cnt[var] += 1
                            for k in self.conn_dict[uid][var]:
                                if k in ['pre','variable']: continue
                                if k not in data[var]: data[var][k] = []
                                data[var][k].append(self.conn_dict[uid][var][k][i])
                            l = len(pre[var])
                            assert(all([len(data[var][k])==l for k in data[var].keys()]))
                    for var,c in cnt.items():
                        npre[var].append(cnt[var])
                else:
                    for n in npre.values(): n.append(0)
            cumpre = {var: np.cumsum([0]+n) for var, n in npre.items()}

            attribs['pre'] = pre
            attribs['cumpre'] = cumpre
            attribs['npre'] = npre
            attribs['conn_data'] = data

    def post_run(self):
        super(LPU, self).post_run()
        for comp in self.components.values():
            comp.post_run()
        # Cycle through IO processors as well
        for p in self.input_processors: p.post_run()
        for p in self.output_processors: p.post_run()

    def run_step(self):
        super(LPU, self).run_step()


        # Update input ports
        self._read_LPU_input()


        # Fetch updated input if available from all input processors
        for p in self.input_processors: p.run_step()

        for model in self.exec_order:
            if model in self.model_var_inj:
                for var in self.model_var_inj[model]:
                    # Reset memory for external input to zero if present
                    self.memory_manager.fill_zeros(model='Input', variable=var)
                    for p in self.input_processors:
                        p.inject_input(var)

        # Call run_step of components
        for model in self.exec_order:
            # Get correct position in buffer for update
            update_pointers = {}
            for var in self._comps[model]['updates']:
                buff = self.memory_manager.get_buffer(var)
                mind = self.memory_manager.variables[var]['models'].index(model)
                shift = self.memory_manager.variables[var]['cumlen'][mind]
                buffer_current_plus_one = buff.current + 1
                if buffer_current_plus_one >= buff.buffer_length:
                    buffer_current_plus_one = 0
                update_pointers[var] = int(buff.gpudata)+\
                                       (buffer_current_plus_one*buff.ld+\
                                        shift)*buff.dtype.itemsize
            self.components[model].run_step(update_pointers)

        # Process output processors
        for p in self.output_processors: p.run_step()

        # Check for transforms

        # Update output ports
        self._extract_output()

        # Step through buffers
        self.memory_manager.step()

        self.time += self.dt

        # Instruct Control inteface to process any pending commands
        if self.control_inteface: self.control_inteface.process_commands()

    def _read_LPU_input(self):
        """
        Extract membrane voltages/spike states from LPU's port map data arrays and
        store them in buffers.
        """
        for var in self.port_inds_gpot.keys():
            # Get correct position in buffer for update
            buff = self.memory_manager.get_buffer(var)
            dest_mem = garray.GPUArray((1,buff.size),buff.dtype,
                                       gpudata=int(buff.gpudata)+\
                                       buff.current*buff.ld*\
                                       buff.dtype.itemsize)
            self.set_inds_both(self.pm['gpot'].data, dest_mem,
                               self.port_inds_gpot[var],self.var_inds_gpot[var])
        for var in self.port_inds_spk.keys():
            # Get correct position in buffer for update
            buff = self.memory_manager.get_buffer(var)
            dest_mem = garray.GPUArray((1,buff.size),buff.dtype,
                                       gpudata=int(buff.gpudata)+\
                                       buff.current*buff.ld*\
                                       buff.dtype.itemsize)
            self.set_inds_both(self.pm['spike'].data, dest_mem, \
                          self.port_inds_spk[var],self.var_inds_spk[var])

    def _extract_output(self):
        """
        Extract membrane voltages/spike states from LPU's port map data arrays and
        store them in buffers.
        """

        for var in self.out_port_inds_gpot.keys():
            # Get correct position in buffer for update
            buff = self.memory_manager.get_buffer(var)
            src_mem = garray.GPUArray((1,buff.size),buff.dtype,
                                       gpudata=int(buff.gpudata)+\
                                       buff.current*buff.ld*\
                                      buff.dtype.itemsize)
            self.set_inds_both(src_mem, self.pm['gpot'].data, \
                    self.out_var_inds_gpot[var], self.out_port_inds_gpot[var])
        for var in self.out_port_inds_spk.keys():
            # Get correct position in buffer for update
            buff = self.memory_manager.get_buffer(var)
            src_mem = garray.GPUArray((1,buff.size),buff.dtype,
                                       gpudata=int(buff.gpudata)+\
                                       buff.current*buff.ld*\
                                       buff.dtype.itemsize)
            self.set_inds_both(src_mem, self.pm['spike'].data, \
                    self.out_var_inds_spk[var], self.out_port_inds_spk[var])

    def set_inds_both(self, src, dest, src_inds, dest_inds):
        """
        Set `dest[dest_inds[i]] = src[src_inds[i]] for i in range(len(src_inds))`
        """

        try:
            func = self.set_inds_both.cache[(src_inds.dtype, src.dtype)]
        except KeyError:
            inds_ctype = dtype_to_ctype(src_inds.dtype)
            data_ctype = dtype_to_ctype(src.dtype)
            v = ("{data_ctype} *dest, {inds_ctype} *dest_inds, " +\
                 "{inds_ctype} *src_inds, {data_ctype} *src").format(\
                        data_ctype=data_ctype,inds_ctype=inds_ctype)
            func = elementwise.ElementwiseKernel(v,\
                            "dest[dest_inds[i]] = src[src_inds[i]]")
            self.set_inds_both.cache[(src_inds.dtype, src.dtype)] = func
        func(dest, dest_inds, src_inds, src, range=slice(0, len(src_inds), 1) )

    set_inds_both.cache = {}

    def _instantiate_component(self, comp_name):
        try:
            cls = self._comps[comp_name]['cls']
        except:
            self.log_info("Error instantiating component of model '%s'" \
                          % comp_name)
            return None

        params_dict = self.memory_manager.parameters[comp_name]
        access_buffers = {var:self.memory_manager.get_buffer(var) \
                          for var in self._comps[comp_name]['accesses'] \
                          if var in self.memory_manager.variables}
        return cls(params_dict, access_buffers, self.dt,
                   LPU_id=self.LPU_id, debug=self.debug,
                   cuda_verbose=bool(self.compile_options))


    def _load_components(self, extra_comps=[]):
        """
        Load all available NDcomponents
        """
        child_classes = NDComponent.NDComponent.__subclasses__()
        comp_classes = child_classes[:]
        for cls in child_classes:
            comp_classes.extend(cls.__subclasses__())
        comp_classes.extend(extra_comps)
        self._comps = {cls.__name__:{'accesses': cls.accesses ,
                                     'updates':cls.updates,
                                     'cls':cls} \
                       for cls in comp_classes if not cls.__name__[:4]=='Base'}
