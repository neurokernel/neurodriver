#!/usr/bin/env python

from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from neurokernel.LPU.NDComponents.NDComponent import NDComponent

class BaseSynapseModel(NDComponent):
    __metaclass__ = ABCMeta

    accesses = ['V']
    updates = ['g']

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
                         (self.num_comps-1) / 256 + 1), 1)
        return func

    def retrieve_buffer_multi(self, param, st = None):
        self.retrieve_buffer_funcs[param].prepared_async_call(
            self.retrieve_buffer_funcs[param].grid,
            self.retrieve_buffer_funcs[param].block,
            st,
            self.access_buffers[param].gpudata,
            self.access_buffers[param].ld,
            self.access_buffers[param].current,
            self.access_buffers[param].buffer_length,
            self.params_dict['pre'][param].gpudata,
            self.params_dict['npre'][param].gpudata,
            self.params_dict['cumpre'][param].gpudata,
            self.params_dict['conn_data'][param]['delay'].gpudata,
            self.params_dict['conn_data'][param]['order'].gpudata,
            self.N_inputs[param],
            self.inputs[param].gpudata,
            self.num_comps)

    def get_retrieve_buffer_multi_func(self, param, dtype):
        template = """
#include "stdio.h"

__global__ void retrieve(%(type)s* buffer, int buffer_ld, int current,
                     int buffer_length, int* pre, int* npre,
                     int* cumpre, int* delay, double* order, int N_inputs,
                     %(type)s* linearized_buffer, int n_items)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    int dl, col, num_pre, start, ordering;

    for(int i = tid; i < n_items; i += total_threads)
    {
        num_pre = npre[i];
        start = cumpre[i];

        if(num_pre == N_inputs)
        {
            for(int j = 0; j < N_inputs; ++j)
            {
                dl = delay[start+j];
                col = current - dl;
                if(col < 0)
                {
                    col = buffer_length + col;
                }
                ordering = order[start+j]; //cast to integer
                linearized_buffer[ordering*n_items+i] = buffer[col*buffer_ld + pre[start+j]];
            }
        }else{
            printf("Warning: npre is not equal to N_inputs.\\n");
        }

    }
}
"""
        mod = SourceModule(template % {"type":dtype_to_ctype(dtype)},
                           options=self.compile_options)
        func = mod.get_function("retrieve")
        func.prepare('PiiiPPPPPiPi')
        func.block = (256,1,1)
        func.grid = (min(6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT,
                         (self.num_comps-1) / 256 + 1), 1)
        return func
