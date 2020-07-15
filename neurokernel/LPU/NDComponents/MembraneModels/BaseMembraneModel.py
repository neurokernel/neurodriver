#!/usr/bin/env python
from abc import ABCMeta, abstractmethod, abstractproperty
from future.utils import with_metaclass
from neurokernel.LPU.NDComponents.NDComponent import NDComponent
from neurokernel.LPU.utils.simpleio import *

from collections import OrderedDict

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


class BaseMembraneModel(with_metaclass(ABCMeta, NDComponent)):
    # __metaclass__ = ABCMeta

    accesses = ['I']
    updates = ['V']
    extra_params = []
    params = []
    internals = OrderedDict([('V', -65.)])

    def __init__(self, params_dict, access_buffers, dt,
                 debug=False, LPU_id=None, cuda_verbose=True):
        if cuda_verbose:
            self.compile_options = ['--ptxas-options=-v']
        else:
            self.compile_options = []

        self.num_comps = params_dict[self.params[0]].size
        self.params_dict = params_dict
        self.access_buffers = access_buffers

        self.debug = debug
        self.LPU_id = LPU_id
        self.dtype = params_dict[self.params[0]].dtype

        self.dt = np.double(dt)

        self.internal_states = {
            c: garray.zeros(self.num_comps, dtype = self.dtype)+self.internals[c] \
            for c in self.internals}

        self.inputs = {
            k: garray.empty(self.num_comps, dtype = self.access_buffers[k].dtype)\
            for k in self.accesses}

        dtypes = {'dt': self.dtype}
        dtypes.update({'input_{}'.format(k): self.inputs[k].dtype for k in self.accesses})
        dtypes.update({'param_{}'.format(k): self.params_dict[k].dtype for k in self.params})
        dtypes.update({'internal_{}'.format(k): self.internal_states[k].dtype for k in self.internals})
        dtypes.update({'update_{}'.format(k): self.dtype for k in self.updates})
        self.update_func = self.get_update_func(dtypes)

    def pre_run(self, update_pointers):
        self.add_initializer('initV', 'V', update_pointers)

    def run_step(self, update_pointers, st=None):
        for k in self.inputs:
            self.sum_in_variable(k, self.inputs[k], st=st)

        self.update_func.prepared_async_call(
            self.update_func.grid, self.update_func.block, st,
            self.num_comps, self.internal_dt, self.internal_steps,
            *[self.inputs[k].gpudata for k in self.accesses]+\
            [self.params_dict[k].gpudata for k in self.params]+\
            [self.internal_states[k].gpudata for k in self.internals]+\
            [update_pointers[k] for k in self.updates])

    def get_update_template(self):
        template = ''
        return template

    def get_update_func(self, dtypes):
        type_dict = {k: dtype_to_ctype(dtypes[k]) for k in dtypes}
        type_dict.update({'fletter': 'f' if type_dict['param_{}'.format(self.params[0])] == 'float' else ''})
        mod = SourceModule(self.get_update_template() % type_dict,
                           options=self.compile_options)
        func = mod.get_function("update")
        func.prepare('i'+np.dtype(dtypes['dt']).char+'i'+'P'*(len(type_dict)-2))
        func.block = (128,1,1)
        func.grid = (min(6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT,
                         (self.num_comps-1) // 128 + 1), 1)
        return func
