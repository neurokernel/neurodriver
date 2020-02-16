#!/usr/bin/env python
"""
Local Processing Unit (LPU) with plugin support for various neuron/synapse models.
"""
import typing as tp
import dataclasses
import time
import collections
import numbers
import copy
import itertools
from types import *
from collections import Counter

from future.utils import iteritems
from past.builtins import long
from builtins import zip

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.elementwise as elementwise

import numpy as np
import networkx as nx

from neurokernel.mixins import LoggerMixin
from neurokernel.core_gpu import Module, CTRL_TAG, GPOT_TAG, SPIKE_TAG
from neurokernel.tools.gpu import get_by_inds
from neurokernel.plsel import Selector


from .utils.simpleio import *
from .utils import parray
from .NDComponents import *
from .MemoryManager import MemoryManager
from . import graph_parser as GParser

import pdb

PORT_IN_GPOT = 'port_in_gpot'
PORT_IN_SPK = 'port_in_spk'
PORT_OUT_GPOT = 'port_out_gpot'
PORT_OUT_SPK = 'port_out_spk'

@dataclasses.dataclass
class LPUConfig:
    id: str = None
    dtype: tp.Union[np.dtype, tp.Any] = np.double
    ctrl_tag: int = CTRL_TAG
    gpot_tag: int = GPOT_TAG
    spike_tag: int = SPIKE_TAG
    device: tp.Union[int, None] = 0
    debug: bool = True
    cuda_verbose: bool = False
    compile_options: tp.List[str] = dataclasses.field(default_factory=list)
    print_timing: bool = False
    time_sync: bool = False

    def __post_init__(self):
        if self.cuda_verbose:
            if '--ptxas-options=-v' not in self.compile_options:
                self.compile_options += ['--ptxas-options=-v']
        else:
            if '--ptxas-options=-v' in self.compile_options:
                self.compile_options.remove('--ptxas-options=-v')


class LPU(Module):
    def __init__(self, dt, comp_dict, conn_list, device=0, input_processors=[],
                 output_processors=[], ctrl_tag=CTRL_TAG, gpot_tag=GPOT_TAG,
                 spike_tag=SPIKE_TAG, rank_to_id=None, routing_table=None,
                 uid_key='id', debug=False, columns=['io', 'type', 'interface'],
                 cuda_verbose=False, time_sync=False, default_dtype=np.double,
                 control_interface=None, id=None, extra_comps=[],
                 print_timing=False):
        """
        :param dt: time step
        :param comp_dict: DictItem of components and attributes. 
            Attributes of all nodes of the same class are collected in a single vector format
        :param conn_list: List of presynaptic, postsynaptic and data in between
            [(pre:str, post:str, data:dict)]
        :param device: GPU device to use.
        :param input_processors: list of input processors
        :param output_processors: list of output processors
        :param ctrl_tag:
            MPI tags that identify messages containing control data values transmitted to
            worker nodes.
        :param gpot_tag:
            MPI tags that identify messages containing
            graded potential port values transmitted to
            worker nodes.
        :param spike_tag:
            MPI tags that identify messages containing spiking port values transmitted to
            worker nodes.
        :param rank_to_id: bidict.bidict
            Mapping between MPI Ranks and module object IDs
        :param routing_table:
        :param uid_key: 
        :param debug:
        :param columns: interface port attributes, network port for controlling the module instance.
        :param cuda_verbose:
        :param time_sync: Time synchronization flag.
            when True, debug messages are not emitted during module synchronization and the
            time taken to receive all incoming data is computed
        :param default_dtype:
        :param control_interface:
        :param id: str Module identifier
        :param extra_comps:
        :param print_timing:
        """
        LoggerMixin.__init__(self, 'LPU {}'.format(id))

        assert('io' in columns)
        assert('type' in columns)
        assert('interface' in columns)
        self.dt = dt
        self.cfg = LPUConfig(
            id=id,
            dtype=default_dtype,
            ctrl_tag=ctrl_tag,
            gpot_tag=gpot_tag,
            spike_tag=spike_tag,
            device=device,
            debug=debug,
            cuda_verbose=cuda_verbose,
            print_timing=print_timing,
            time_sync=time_sync
        )
        self.time = 0
        self.control_interface = control_interface

        # input output processors
        if not isinstance(input_processors, list):
            input_processors = [input_processors]
        if not isinstance(output_processors, list):
            input_processors = [output_processors]
        self.output_processors = output_processors
        self.input_processors = input_processors

        self._uid_key = uid_key
        self._uid_generator = uid_generator()

        # Load all NDComponents Class
        self._comps = self._load_components(extra_comps=extra_comps)
        
        # instantiated componnets, instantiated at `self.pre_run`
        self.components = {}

        # memory_manager created at `pre_run`
        self.memory_manager = None

        # Ignore models without implementation
        models_to_be_deleted = []
        for model in comp_dict:
            if not model in self._comps and not model in ['Port','Input']:
                self.log_info("Ignoring Model %s: Can not find implementation"
                              % model)
                models_to_be_deleted.append(model)
        for model in models_to_be_deleted:
            del comp_dict[model]

        # timing each part of parsing
        if self.cfg.print_timing:
            start = time.time()

        # Delay for each variable. Assume zero delay by default
        self.variable_delay_map = {}

        # Generate a uid to model map of components
        self.uid_model_map = {}
        for model,attribs in comp_dict.items():
            for i,uid in enumerate(attribs[uid_key]):
                self.uid_model_map[uid] = model

        if self.cfg.print_timing:
            self.log_info("Elapsed time for processing comp_dict: {:.3f} seconds".format(time.time()-start))
            start = time.time()

        # process conn_list, update variable_delay_map as needed
        agg_map, agg, conns, self.in_port_vars, self.out_port_conns = self._process_conn_list(comp_dict, conn_list)
        if self.cfg.print_timing:
            self.log_info("Elapsed time for processing conn_list: {:.3f} seconds".format(time.time()-start))
            start = time.time()

        if agg and not 'Aggregator' in comp_dict:
            comp_dict['Aggregator'] = {uid_key: []}
        if agg:
            agg_uid_key_set = set(comp_dict['Aggregator'][uid_key])

        # Add updated aggregator components to component dictionary
        # and create connections for aggregator
        for post, conn_list in agg.items():
            uid = agg_map[post]
            if uid not in agg_uid_key_set:
                keys = [k for k in comp_dict['Aggregator'] if k != uid_key]
                comp_dict['Aggregator'][uid_key].append(uid)
                agg_uid_key_set.add(uid)
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
        if self.cfg.print_timing:
            self.log_info("Elapsed time for processing aggregator: {:.3f} seconds".format(time.time()-start))
            start = time.time()

        self.conn_dict = {}
        # RePackage connections
        for (pre, post, data) in conns:
            if not post in self.conn_dict:
                self.conn_dict[post] = {}
            var = data['variable']
            data.pop('variable')
            if not var in self.conn_dict[post]:
                self.conn_dict[post][var] = {k:[] for k in ['pre'] + list(data)}
            self.conn_dict[post][var]['pre'].append(pre)
            for k in data: self.conn_dict[post][var][k].append(data[k])

        if self.cfg.print_timing:
            self.log_info("Elapsed time for repackaging connections: {:.3f} seconds".format(time.time()-start))
            start = time.time()

        # Add connections for component with no incoming connections
        for uid, model in iteritems(self.uid_model_map):
            if model == 'Port':
                continue
            for var in self._comps[model]['accesses']:
                if ((uid not in self.conn_dict or var not in self.conn_dict[uid])):
                    pre = self._uid_generator.generate_uid(input=True)
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
                        comp_dict['Input'][var] = {self._uid_key: []}
                    comp_dict['Input'][var][self._uid_key].append(pre)

        if self.cfg.print_timing:
            self.log_info("Elapsed time for adding connections for component with no incoming connections: {:.3f} seconds".format(time.time()-start))
            start = time.time()

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
            for k in n:
                n[k] = [n[k][i] for i in order]

        # Reorder input port variables
        for var, uids in self.in_port_vars.items():
            order = np.argsort([self.uid_ind_map['Port'][uid] for uid in uids])
            self.in_port_vars[var] = [uids[i] for i in order]

        if self.cfg.print_timing:
            self.log_info("Elapsed time for optimizing ordering: {:.3f} seconds".format(time.time()-start))

        # Try to figure out order of stepping through components
        # If a loop of dependencies is present, update order behaviour is undefined
        models = list(comp_dict)
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

        # execute models in by dependency
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

        if self.cfg.print_timing:
            start = time.time()
        # Get selectors of input ports:
        sel_in_gpot, self.in_gpot_uids = self.extract_in_gpot(comp_dict,
                                                              self._uid_key)
        self.sel_in_gpot = Selector(','.join(sel_in_gpot))
        sel_in_spk, self.in_spk_uids = self.extract_in_spk(comp_dict,
                                                           self._uid_key)
        self.sel_in_spk = Selector(','.join(sel_in_spk))
        sel_in = Selector.add(self.sel_in_gpot, self.sel_in_spk)

        # Get selectors of output neurons:
        sel_out_gpot, self.out_gpot_uids = self.extract_out_gpot(comp_dict,
                                                                 self._uid_key)
        self.sel_out_gpot = Selector(','.join(sel_out_gpot))
        sel_out_spk, self.out_spk_uids = self.extract_out_spk(comp_dict,
                                                              self._uid_key)
        self.sel_out_spk = Selector(','.join(sel_out_spk))
        sel_out = Selector.add(self.sel_out_gpot, self.sel_out_spk)
        sel_gpot = Selector.add(self.sel_in_gpot, self.sel_out_gpot)
        sel_spk = Selector.add(self.sel_in_spk, self.sel_out_spk)
        sel = Selector.add(sel_gpot, sel_spk)
        if self.cfg.print_timing:
            self.log_info("Elapsed time for generating selectors: {:.3f} seconds".format( time.time()-start))

        # Save component parameters data in the form
        # [('Model0', {'attrib0': [..], 'attrib1': [..]}), ('Model1', ...)]
        self.comp_list = comp_dict.items()
        self.models = {m:i for i,(m,_) in enumerate(self.comp_list)}

        # Number of components of each model:
        self.model_num = [len(n[uid_key]) if not m=='Input' else
                          len(sum([d[uid_key] for d in n.values()],[]))
                          for m, n in self.comp_list]

        data_gpot = np.zeros(len(self.in_gpot_uids)+len(self.out_gpot_uids),
                             self.cfg.dtype)
        data_spike = np.zeros(len(self.in_spk_uids)+len(self.out_spk_uids),
                              self.cfg.dtype)

        if self.cfg.print_timing:
            start = time.time()
        super(LPU, self).__init__(sel=sel, sel_in=sel_in, sel_out=sel_out,
                                  sel_gpot=sel_gpot, sel_spike=sel_spk,
                                  data_gpot=data_gpot, data_spike=data_spike,
                                  columns=columns, ctrl_tag=self.cfg.ctrl_tag, gpot_tag=self.cfg.gpot_tag,
                                  spike_tag=self.cfg.spike_tag, id=self.cfg.id,
                                  rank_to_id=rank_to_id, routing_table=routing_table,
                                  device=self.cfg.device, debug=self.cfg.debug, time_sync=self.cfg.time_sync,
                                  print_timing=self.cfg.print_timing)

        if self.cfg.print_timing:
            cuda.Context.synchronize()
            self.log_info("Elapsed time for initializing parent class: {:.3f} seconds".format(time.time()-start))


        # Integer indices in port map data arrays corresponding to input/output
        # gpot/spiking ports:
        self.in_gpot_inds = np.array(self.pm['gpot'].ports_to_inds(
                                     self.sel_in_gpot), dtype=np.int32)
        self.out_gpot_inds = np.array(self.pm['gpot'].ports_to_inds(
                                      self.sel_out_gpot), dtype=np.int32)
        self.in_spk_inds = np.array(self.pm['spike'].ports_to_inds(
                                    self.sel_in_spk), dtype=np.int32)
        self.out_spk_inds = np.array(self.pm['spike'].ports_to_inds(
                                     self.sel_out_spk), dtype=np.int32)

    def pre_run(self):
        '''Pre Run
        
        1. super().pre_run() see `neurokernel.core_gpu.Module.pre_run`
        2. instantiate MemoryManager
        3. initialize variable memory
        4. process_connections
        5. initialize parameters
        6. instantiate components
        7. initialize Port I/O
        8. intiialize I/O Processors
        '''
        if self.cfg.print_timing:
            start = time.time()
        super(LPU, self).pre_run()
        if self.cfg.print_timing:
            start = time.time()
            self.log_info("LPU pre_run parent took {} seconds".format(time.time()-start))

        if self.cfg.print_timing:
            start = time.time()
        self.memory_manager = MemoryManager()
        self._init_variable_memory()
        if self.cfg.print_timing:
            cuda.Context.synchronize()
            self.log_info('Elapsed time for initialing variable memory: {:.3f} seconds'.format( time.time()-start))
            start = time.time()
        self._process_connections()
        if self.cfg.print_timing:
            self.log_info('Elapsed time for process_connections: {:.3f} seconds'.format(time.time()-start))
            start = time.time()
        self._init_parameters()
        if self.cfg.print_timing:
            cuda.Context.synchronize()
            self.log_info('Elapsed time for init_paramseters: {:.3f} seconds'.format( time.time()-start))
            start = time.time()

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
                            int(int(buff.gpudata)+(j*buff.ld+\
                                                shift)*buff.dtype.itemsize),
                            int(int(buff.gpudata)+(buff.current*buff.ld+\
                                                shift)*buff.dtype.itemsize),
                            int(buff.dtype.itemsize*self.model_num[self.models[model]]))
        if self.cfg.print_timing:
            cuda.Context.synchronize()
            self.log_info('Elapsed time for instantiating components: {:.3f} seconds'.format(time.time()-start))
            start = time.time()
        # Setup ports
        self._setup_input_ports()
        self._setup_output_ports()

        if self.cfg.print_timing:
            cuda.Context.synchronize()
            self.log_info('Elapsed time for setting up ports: {:.3f} seconds'.format( time.time()-start))
            start = time.time()

        for p in self.input_processors:
            p.LPU_obj = self
            p._pre_run()

        for p in self.output_processors:
            p.LPU_obj = self
            p._pre_run()

        if self.cfg.print_timing:
            cuda.Context.synchronize()
            self.log_info('Elapsed time for prerun input and output processors: {:.3f} seconds'.format( time.time()-start))

        self.memory_manager.precompile_fill_zeros()

        if self.control_interface: self.control_interface.register(self)
        if self.cfg.print_timing:
            cuda.Context.synchronize()
            self.log_info("Elapsed time for LPU pre_run: {:.3f} seconds".format(time.time()-start))

        if self.cfg.print_timing:
            self.timing = {'read_input': 0, 'input_processors': 0, 'inject_input': 0,
                           'model_run': 0, 'output_processors': 0,
                           'extract_output': 0, 'total': 0}

    def run_step(self):
        super(LPU, self).run_step()

        # Update input ports
        if self.cfg.print_timing:
            start_all = time.time()
            start = time.time()
        self._read_LPU_input()
        if self.cfg.print_timing:
            cuda.Context.synchronize()
            self.timing['read_input'] += time.time()-start

        # Fetch updated input if available from all input processors
        if self.cfg.print_timing:
            start = time.time()
        for p in self.input_processors: p.run_step()
        if self.cfg.print_timing:
            cuda.Context.synchronize()
            self.timing['input_processors'] += time.time()-start

        if self.cfg.print_timing:
            start = time.time()
        for model in self.exec_order:
            if model in self.model_var_inj:
                for var in self.model_var_inj[model]:
                    # Reset memory for external input to zero if present
                    self.memory_manager.fill_zeros(model='Input', variable=var)
                    for p in self.input_processors:
                        p.inject_input(var)
        if self.cfg.print_timing:
            cuda.Context.synchronize()
            self.timing['inject_input'] += time.time()-start

        # Call run_step of components
        if self.cfg.print_timing:
            start = time.time()
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
        if self.cfg.print_timing:
            cuda.Context.synchronize()
            self.timing['model_run'] += time.time()-start

        # Process output processors
        if self.cfg.print_timing:
            start = time.time()
        for p in self.output_processors: p.run_step()
        if self.cfg.print_timing:
            cuda.Context.synchronize()
            self.timing['output_processors'] += time.time()-start

        # Update output ports
        if self.cfg.print_timing:
            start = time.time()
        self._extract_output()
        if self.cfg.print_timing:
            cuda.Context.synchronize()
            self.timing['extract_output'] += time.time()-start

        # Step through buffers
        self.memory_manager.step()

        self.time += self.dt

        # Instruct Control inteface to process any pending commands
        if self.control_interface: self.control_interface.process_commands()
        if self.cfg.print_timing:
            cuda.Context.synchronize()
            self.timing['total'] += time.time()-start_all

    def post_run(self):
        super(LPU, self).post_run()
        for comp in self.components.values():
            comp.post_run()
        # Cycle through IO processors as well
        for p in self.input_processors: p.post_run()
        for p in self.output_processors: p.post_run()
        if self.cfg.print_timing:
            print('time spent on:', self.timing)

    def _process_conn_list(self, comp_dict, conn_list,
                           reverse_set=('reverse','Vr','VR','reverse_potential')):
        '''Process Connections between components

        remove inconsitent connections
        calculate required delays, infer variable if required
        '''
        in_port_vars_set = {}

        # Map from post synaptic component to aggregator uid
        agg_map = {}
        agg = {}
        conns = []
        in_port_vars = {}
        out_port_conns = []
        comp_uid_order = {}
        for model, attribs in comp_dict.items():
            comp_uid_order[model] = {uid: i for i, uid in enumerate(attribs[self._uid_key])}

        comp_updates = {name: v['updates'] if not name=='Port' else [] \
                        for name, v in self._comps.items()}
        comp_updates.update({'Port': []})
        comp_accesses = {name: v['accesses'] if not name=='Port' else [] \
                         for name, v in self._comps.items()}
        comp_accesses.update({'Port': []})

        for pre, post, data in conn_list:
            try:
                pre_model = self.uid_model_map[pre]
            except KeyError:
                continue
            try:
                post_model = self.uid_model_map[post]
            except KeyError:
                continue

            # Update delay, overwritting `data` dict 
            if 'delay' in data:
                _delay = int(round((data['delay']/self.dt)))
            else:
                _delay = 0
            delay = max(_delay, 1) - 1
            data['delay'] = delay

            pre_updates = comp_updates[pre_model]
            pre_updates_set = set(pre_updates)
            post_accesses = comp_accesses[post_model]
            post_accesses_set = set(post_accesses)
            pre_post_intersection = pre_updates_set & post_accesses_set
            g_in_pre_update = 'g' in pre_updates_set # not so useful
            V_in_pre_update = 'V' in pre_updates_set

            if post_model == 'Aggregator':
                agg_map[post] = post
                # check if reverse potential exist from list of allowed keywords
                reverse_key = None
                for k in reverse_set:
                    if k in data:
                        reverse_key = k
                        reverse = data[reverse_key]
                        break
                # If no key in data, look in the attibutes of the synapse
                if reverse_key is None:
                    s = set(reverse_set) & set(comp_dict[pre_model])
                    if s: reverse_key = s.pop()
                    if reverse_key:
                        reverse = comp_dict[pre_model][reverse_key][\
                                comp_uid_order[pre_model][pre]]
                        if g_in_pre_update:
                            data['reverse'] = reverse
                    else:
                        if g_in_pre_update:
                            self.log_info('Assuming reverse potential ' +
                                          'to be zero for connection from' +
                                          '%s to %s'%(pre,post))
                            data['reverse'] = 0
                        reverse = 0

                if post in agg:
                    if g_in_pre_update:
                        agg[post].append({'pre':pre,'reverse':reverse,
                                          'variable':'g'})
                    elif V_in_pre_update:
                        agg[post].append({'pre':pre, 'variable':'V'})
                else:
                    # Make sure aggregator has access to postsynaptic voltage
                    if g_in_pre_update:
                        agg[post] = [{'pre':pre,'reverse':reverse,'variable':'g'}]
                    elif V_in_pre_update:
                        agg[post] = [{'pre':pre,'variable':'V'}]

                if g_in_pre_update:
                    agg[post][-1].update(data)
                    self.variable_delay_map['g'] = max(data['delay'],
                                    self.variable_delay_map['g'] if 'g' in \
                                    self.variable_delay_map else 0)

            # Insert Aggregator between g->V if required. Assume 'reverse' or
            # 'Vr' or 'VR' or 'reverse_potential' id present as a param in the
            # synapse in that case
            if not pre_post_intersection:
                if g_in_pre_update and 'I' in post_accesses_set:
                    # start2 = time.time()
                    # First look for reverse in the attributes of the edge
                    reverse_key = None
                    for k in reverse_set:
                        if k in data:
                            reverse_key = k
                            reverse = data[reverse_key]
                            break
                    if reverse_key is None:
                        # else look in the attibutes of the synapse
                        s = (set(['reverse','Vr','VR','reverse_potential'])&
                             set(comp_dict[pre_model]))
                        if s: reverse_key = s.pop()
                        if reverse_key:
                            reverse = comp_dict[pre_model][reverse_key][\
                                    comp_uid_order[pre_model][pre]]
                        else:
                            self.log_info('Assuming reverse potential ' +
                                          'to be zero for connection from' +
                                          '%s to %s'%(pre,post))
                            reverse = 0

                    data.update({'pre':pre,'reverse':reverse,
                                 'variable':'g'})
                    if post in agg:
                        agg[post].append(data)
                    else:
                        # Make sure aggregator has access to postsynaptic voltage
                        agg[post] = [{'pre':post,'variable':'V'},
                                     data]
                    if post not in agg_map:
                        uid = self._uid_generator.generate_uid()
                        agg_map[post] = uid
                    self.variable_delay_map['g'] = max(data['delay'],
                                    self.variable_delay_map['g'] if 'g' in \
                                    self.variable_delay_map else 0)

                elif pre_model == 'Port':
                    if not 'variable' in data:
                        data['variable'] = post_accesses[0]
                    if not data['variable'] in in_port_vars:
                        in_port_vars[data['variable']] = []
                        in_port_vars_set[data['variable']] = set()
                    if pre not in in_port_vars_set[data['variable']]:
                        in_port_vars[data['variable']].append(pre)
                        in_port_vars_set[data['variable']].add(pre)
                    conns.append((pre, post, data))
                    self.variable_delay_map[data['variable']] = max(data['delay'],
                            self.variable_delay_map[data['variable']] if \
                            data['variable'] in self.variable_delay_map else 0)
                elif post_model == 'Port':
                    if not 'variable' in data:
                        data['variable'] = pre_updates[0]
                    out_port_conns.append((pre, post, data['variable']))
                else:
                    self.log_info("Ignoring connection %s -> %s"%(pre,post))
                continue

            var = data['variable'] if 'variable' in data else None
            if not var:
                var = pre_post_intersection.pop()
            elif not (var in pre_updates_set and var in post_accesses_set):
                continue
            data['variable'] = var
            self.variable_delay_map[data['variable']] = max(data['delay'],
                            self.variable_delay_map[data['variable']] if \
                            data['variable'] in self.variable_delay_map else 0)
            # connection to Aggregator will be added later
            if 'Aggregator' not in post_model:
                conns.append((pre,post,data))
        return agg_map, agg, conns, in_port_vars, out_port_conns
    
    def _setup_output_ports(self):
        ''' Setup output ports
        TODO: optimize the order of self.out_port_conns beforehand
        '''
        self.out_port_inds_gpot = {}
        self.out_var_inds_gpot = {}
        self.out_port_inds_spk = {}
        self.out_var_inds_spk = {}
        # assuming that the UIDs are unique
        out_gpot_index = {uid: i for i, uid in enumerate(self.out_gpot_uids)}
        out_spk_index = {uid: i for i, uid in enumerate(self.out_spk_uids)}
        for pre_uid, post_uid, var in self.out_port_conns:
            if not var in self.out_port_inds_gpot:
                self.out_port_inds_gpot[var] = []
                self.out_var_inds_gpot[var] = []
                self.out_port_inds_spk[var] = []
                self.out_var_inds_spk[var] = []
            ind = self.memory_manager.variables[var]['uids'][pre_uid]
            if post_uid in out_gpot_index:
                # self.out_port_inds_gpot[var].append(self.out_gpot_inds[\
                #                             self.out_gpot_uids.index(post_uid)])
                self.out_port_inds_gpot[var].append(self.out_gpot_inds[\
                                            out_gpot_index[post_uid]])
                self.out_var_inds_gpot[var].append(ind)
            else:
                self.out_port_inds_spk[var].append(self.out_spk_inds[\
                                            out_spk_index[post_uid]])
                self.out_var_inds_spk[var].append(ind)

        tmp = self.out_port_inds_gpot.copy()
        for var in tmp:
            if not self.out_port_inds_gpot[var]:
                del self.out_port_inds_gpot[var]
                del self.out_var_inds_gpot[var]
            else:
                self.out_port_inds_gpot[var] = garray.to_gpu(\
                        np.array(self.out_port_inds_gpot[var],np.int32))
                self.out_var_inds_gpot[var] = garray.to_gpu(\
                        np.array(self.out_var_inds_gpot[var],np.int32))

        tmp = self.out_port_inds_spk.copy()
        for var in tmp:
            if not self.out_port_inds_spk[var]:
                del self.out_port_inds_spk[var]
                del self.out_var_inds_spk[var]
            else:
                self.out_port_inds_spk[var] = garray.to_gpu(\
                        np.array(self.out_port_inds_spk[var],np.int32))
                self.out_var_inds_spk[var] = garray.to_gpu(\
                        np.array(self.out_var_inds_spk[var],np.int32))

    def _setup_input_ports(self):
        '''setup input ports
        '''
        self.port_inds_gpot = {}
        self.var_inds_gpot = {}
        self.port_inds_spk = {}
        self.var_inds_spk = {}
        # assuming that the UIDs are unique
        in_gpot_index = {uid: i for i, uid in enumerate(self.in_gpot_uids)}
        in_spk_index = {uid: i for i, uid in enumerate(self.in_spk_uids)}
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
                if uid in in_gpot_index:
                    self.port_inds_gpot[var].append(self.in_gpot_inds[\
                                            in_gpot_index[uid]])
                    self.var_inds_gpot[var].append(i + shift)
                else:
                    self.port_inds_spk[var].append(self.in_spk_inds[\
                                            in_spk_index[uid]])
                    self.var_inds_spk[var].append(i + shift)
        tmp = self.port_inds_gpot.copy()
        for var in tmp:
            if not self.port_inds_gpot[var]:
                del self.port_inds_gpot[var]
                del self.var_inds_gpot[var]
            else:
                self.port_inds_gpot[var] = garray.to_gpu(\
                        np.array(self.port_inds_gpot[var],np.int32))
                self.var_inds_gpot[var] = garray.to_gpu(\
                        np.array(self.var_inds_gpot[var],np.int32))
        tmp = self.port_inds_spk.copy()
        for var in tmp:
            if not self.port_inds_spk[var]:
                del self.port_inds_spk[var]
                del self.var_inds_spk[var]
            else:
                self.port_inds_spk[var] = garray.to_gpu(\
                        np.array(self.port_inds_spk[var],np.int32))
                self.var_inds_spk[var] = garray.to_gpu(\
                        np.array(self.var_inds_spk[var],np.int32))


    def _init_parameters(self):
        for m, n in self.comp_list:
            if not m in ['Port','Input']:
                nn = n.copy()
                nn.pop(self._uid_key)
                # copy integer and boolean parameters into separate dictionary
                nn_int = {k:v for k, v in iteritems(nn) if (isinstance(v, list)
                            and len(v) and type(v[0]) in [int, bool])}
                nn_rest = {k:v for k, v in iteritems(nn) if (
                           (not isinstance(v, list)) or (len(v) and
                           type(v[0]) not in [int, long, bool]))}
                if nn_int:
                    self.memory_manager.params_htod(m, nn_int, np.int32)
                if nn_rest:
                    self.memory_manager.params_htod(m, nn_rest,
                                                    self.cfg.dtype)

    def _init_variable_memory(self):
        '''Initialize Memory for all Variables
        '''
        var_info = {}
        for (model, attribs) in self.comp_list:
            if model in ['Port']: continue
            # Add memory for external inputs if required
            if model == 'Input':
                for var, d in iteritems(attribs):
                    if not var in var_info:
                        var_info[var] = {'models':[],'len':[],'delay':0,'uids':[]}
                    var_info[var]['models'].append('Input')
                    var_info[var]['len'].append(len(d[self._uid_key]))
                    var_info[var]['uids'].extend(d[self._uid_key])
                continue
            for var in self._comps[model]['updates']:
                if not var in var_info:
                    var_info[var] = {'models':[],'len':[],'delay':0,'uids':[]}
                var_info[var]['models'].append(model)
                var_info[var]['len'].append(len(attribs[self._uid_key]))
                var_info[var]['uids'].extend(attribs[self._uid_key])

        # Add memory for input ports
        for var in self.in_port_vars:
            if not var in var_info:
                var_info[var] = {'models':[],'len':[],'delay':0,'uids':[]}
            var_info[var]['models'].append('Port')
            var_info[var]['len'].append(len(self.in_port_vars[var]))
            var_info[var]['uids'].extend(self.in_port_vars[var])

        for var in self.variable_delay_map:
            var_info[var]['delay'] = self.variable_delay_map[var]

        for var, d in var_info.items():
            d['cumlen'] = np.cumsum([0]+d['len'])
            d['uids'] = {uid:i for i, uid in enumerate(d['uids'])}
            self.memory_manager.memory_alloc(var, d['cumlen'][-1], d['delay']+2,\
                dtype=self.cfg.dtype,
                info=d)

    def _process_connections(self):
        '''Process Connections and update `comp_list`
        4 attributes are computed and updated for each component:
        1. pre: 
        2. cumpre:
        3. npre:
        4. conn_data:
        '''
        for (model, attribs) in self.comp_list:
            if model in ['Port', 'Input']:
                continue
            pre = {var:[] for var in self._comps[model]['accesses']}
            npre = {var:[] for var in self._comps[model]['accesses']}
            data = {var:{} for var in self._comps[model]['accesses']}
            for uid in attribs[self._uid_key]:
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
                            assert(all([len(data[var][k])==l for k in data[var]]))
                    for var,c in cnt.items():
                        npre[var].append(cnt[var])
                else:
                    for n in npre.values(): n.append(0)
            cumpre = {var: np.cumsum([0]+n) for var, n in npre.items()}

            attribs['pre'] = pre
            attribs['cumpre'] = cumpre
            attribs['npre'] = npre
            attribs['conn_data'] = data

    def _read_LPU_input(self):
        """
        Extract membrane voltages/spike states from LPU's port map data arrays and
        store them in buffers.
        """
        for var in self.port_inds_gpot:
            # Get correct position in buffer for update
            buff = self.memory_manager.get_buffer(var)
            dest_mem = garray.GPUArray((1,buff.size),buff.dtype,
                                       gpudata=int(buff.gpudata)+\
                                       buff.current*buff.ld*\
                                       buff.dtype.itemsize)
            self.set_inds_both(self.pm['gpot'].data, dest_mem,
                               self.port_inds_gpot[var],self.var_inds_gpot[var])
        for var in self.port_inds_spk:
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

        for var in self.out_port_inds_gpot:
            # Get correct position in buffer for update
            buff = self.memory_manager.get_buffer(var)
            src_mem = garray.GPUArray((1,buff.size),buff.dtype,
                                       gpudata=int(buff.gpudata)+\
                                       buff.current*buff.ld*\
                                      buff.dtype.itemsize)
            self.set_inds_both(src_mem, self.pm['gpot'].data, \
                    self.out_var_inds_gpot[var], self.out_port_inds_gpot[var])
        for var in self.out_port_inds_spk:
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
        """Instantiate Component by name"""
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
                   LPU_id=self.cfg.id, debug=self.cfg.debug,
                   cuda_verbose=bool(self.cfg.compile_options))


    def _load_components(self, extra_comps=[]):
        """Load all available NDcomponents"""
        child_classes = NDComponent.NDComponent.__subclasses__()
        comp_classes = child_classes[:]
        for cls in child_classes:
            comp_classes.extend(cls.__subclasses__())
        comp_classes.extend(extra_comps)
        return {cls.__name__:{'accesses': cls.accesses ,
                              'updates':cls.updates,
                              'cls':cls} \
                for cls in comp_classes if not cls.__name__[:4]=='Base'}

    @staticmethod
    def lpu_parser(filename):
        return GParser.lpu_parser(filename)

    @staticmethod
    def graph_to_dicts(graph, uid_key=None, class_key='class', remove_edge_id=False):
        return GParser.graph_to_dicts(graph, uid_key, class_key, remove_edge_id)

    @classmethod
    def extract_in_gpot(cls, comp_dict, uid_key):
        return GParser.extract_in_gpot(comp_dict, uid_key)

    @classmethod
    def extract_in_spk(cls, comp_dict, uid_key):
        return GParser.extract_in_spk(comp_dict, uid_key)

    @classmethod
    def extract_out_gpot(cls, comp_dict, uid_key):
        return GParser.extract_out_gpot(comp_dict, uid_key)

    @classmethod
    def extract_out_spk(cls, comp_dict, uid_key):
        return GParser.extract_out_spk(comp_dict, uid_key)

    @classmethod
    def extract_sel_in_gpot(cls, comp_dict):
        return GParser.extract_sel_in_gpot(comp_dict)

    @classmethod
    def extract_sel_in_spk(cls, comp_dict):
        return GParser.extract_sel_in_spk(comp_dict)

    @classmethod
    def extract_sel_out_gpot(cls, comp_dict):
        return GParser.extract_sel_out_gpot(comp_dict)

    @classmethod
    def extract_sel_out_spk(cls, comp_dict):
        return GParser.extract_sel_out_spk(comp_dict)

    @classmethod
    def extract_sel_in(cls, comp_dict):
        return GParser.extract_sel_in(comp_dict)

    @classmethod
    def extract_sel_out(cls, comp_dict):
        return GParser.extract_sel_out(comp_dict)


class uid_generator(object):
    def __init__(self):
        self.input_count = 0
        self.auto_count = 0

    def generate_uid(self, input=False):
        if input:
            uid = 'input_' + str(self.input_count)
            self.input_count += 1
        else:
            uid = 'auto_' + str(self.auto_count)
            self.auto_count += 1
        return uid
