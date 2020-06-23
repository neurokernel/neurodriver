#!/usr/bin/env python

from collections import OrderedDict

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from abc import ABCMeta, abstractmethod, abstractproperty
from future.utils import with_metaclass

from neurokernel.LPU.NDComponents.NDComponent import NDComponent

class BaseSynapseModel(with_metaclass(ABCMeta, NDComponent)):
    # __metaclass__ = ABCMeta

    accesses = ['V']
    updates = ['g']
    params = []
    extra_params = []
    internals = []

    def __init__(self, params_dict, access_buffers, dt,
                 LPU_id=None, debug=False, cuda_verbose=False):
        if cuda_verbose:
            self.compile_options = ['--ptxas-options=-v']
        else:
            self.compile_options = []

        self.debug = debug
        self.num_comps = params_dict[self.params[0]].size
        self.dtype = params_dict[self.params[0]].dtype
        self.LPU_id = LPU_id
        self.dt = np.double(dt)
        #self.ddt = np.double(1e-6)
        #self.steps = np.int32(max( int(self.dt/self.ddt), 1 ))

        self.params_dict = params_dict
        self.access_buffers = access_buffers

        self.internal_states = {
            c: garray.zeros(self.num_comps, dtype = self.dtype)+self.internals[c] \
            for c in self.internals}

        self.inputs = {
            k: garray.empty(self.num_comps, dtype = self.access_buffers[k].dtype)\
            for k in self.accesses}

        self.retrieve_buffer_funcs = {}
        for k in self.accesses:
            self.retrieve_buffer_funcs[k] = \
                self.get_retrieve_buffer_func(
                    k, dtype = self.access_buffers[k].dtype)

        dtypes = {'dt': self.dtype}
        dtypes.update({'input_{}'.format(k): self.inputs[k].dtype for k in self.accesses})
        dtypes.update({'param_{}'.format(k): self.params_dict[k].dtype for k in self.params})
        dtypes.update({'internal_{}'.format(k): self.internal_states[k].dtype for k in self.internals})
        dtypes.update({'update_{}'.format(k): self.dtype for k in self.updates})
        self.update_func = self.get_update_func(dtypes)

    def retrieve_buffer(self, param, st = None):
        self.retrieve_buffer_funcs[param].prepared_async_call(
            self.retrieve_buffer_funcs[param].grid,
            self.retrieve_buffer_funcs[param].block,
            st,
            self.access_buffers[param].gpudata,
            self.access_buffers[param].ld,                    #i
            self.access_buffers[param].current,               #i
            self.access_buffers[param].buffer_length,         #i
            self.params_dict['pre'][param].gpudata,                #P
            self.params_dict['npre'][param].gpudata,               #P
            self.params_dict['cumpre'][param].gpudata,             #P
            self.params_dict['conn_data'][param]['delay'].gpudata,
            self.inputs[param].gpudata,
            self.num_comps)

    def get_retrieve_buffer_func(self, param, dtype):
        template = """
__global__ void retrieve(%(type)s* buffer, int buffer_ld, int current,
                         int buffer_length, int* pre, int* npre,
                         int* cumpre, int* delay,
                         %(type)s* linearized_buffer, int n_items)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    int dl, col;
    for(int i = tid; i < n_items; i += total_threads)
    {
        // npre[i] here is always 1, so no else is provided.
        if(npre[i])
        {
            dl = delay[i];
            col = current - dl;
            if(col < 0)
            {
                col = buffer_length + col;
            }
            linearized_buffer[i] = buffer[col*buffer_ld + pre[cumpre[i]]];
        }
    }
}
        """
        mod = SourceModule(template % {"type":dtype_to_ctype(dtype)},
                           options=self.compile_options)
        func = mod.get_function("retrieve")
        func.prepare('PiiiPPPPPi')
        func.block = (256,1,1)
        func.grid = (min(6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT,
                         (self.num_comps-1) // 256 + 1), 1)
        return func

    def run_step(self, update_pointers, st = None):
        # retrieve all buffers into a linear array
        for k in self.inputs:
            self.retrieve_buffer(k, st = st)

        self.update_func.prepared_async_call(
            self.update_func.grid, self.update_func.block, st,
            self.num_comps, self.internal_dt, self.internal_steps,
            *[self.inputs[k].gpudata for k in self.accesses]+\
            [self.params_dict[k].gpudata for k in self.params]+\
            [self.internal_states[k].gpudata for k in self.internals]+\
            [update_pointers[k] for k in self.updates])

    def get_update_func(self, dtypes):
        type_dict = {k: dtype_to_ctype(dtypes[k]) for k in dtypes}
        type_dict.update({'fletter': 'f' if type_dict['param_{}'.format(self.params[0])] == 'float' else ''})
        mod = SourceModule(self.get_update_template() % type_dict,
                           options=self.compile_options)
        func = mod.get_function("update")
        func.prepare('i'+np.dtype(dtypes['dt']).char+'i'+'P'*(len(type_dict)-2))
        func.block = (256,1,1)
        func.grid = (min(6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT,
                         (self.num_comps-1) // 256 + 1), 1)
        return func
