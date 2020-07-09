#!/usr/bin/env python


import warnings
from abc import ABCMeta, abstractmethod, abstractproperty
import os.path
import numpy as np
from future.utils import with_metaclass

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from neurokernel.LPU.utils.simpleio import *

class NDComponent(with_metaclass(ABCMeta, object)):
    # __metaclass__ = ABCMeta

    accesses = []
    updates = []

    @abstractmethod
    def __init__(self, params_dict, access_buffers, dt, debug=False,
                 LPU_id=None, cuda_verbose=False):
        pass

    @abstractmethod
    def run_step(self, update_pointers):
        pass

    @property
    @abstractmethod
    def maximum_dt_allowed(self):
        pass

    @property
    def internal_steps(self):
        if self.dt > self.maximum_dt_allowed:
            div = self.dt/self.maximum_dt_allowed
            if np.abs(div - np.round(div)) < 1e-5:
                return int(np.round(div))
            else:
                return int(np.ceil(div))
            #raise ValueError('Simulation time step dt larger than maximum allowed dt of model {}'.format(type(self)))
        else:
            return 1

    @property
    def internal_dt(self):
        return self.dt/self.internal_steps

    def pre_run(self, update_pointers):
        pass

    def post_run(self):
        '''
        This method will be called at the end of the simulation.
        '''
        pass

    def add_initializer(self, var_a, var_b, update_pointers):
        if var_a in self.params_dict:
            if var_b in self.internal_states:
                cuda.memcpy_dtod(self.internal_states[var_b].gpudata,
                                    self.params_dict[var_a].gpudata,
                                    self.params_dict[var_a].nbytes)
            if var_b in update_pointers:
                cuda.memcpy_dtod(int(update_pointers[var_b]),
                                    self.params_dict[var_a].gpudata,
                                    self.params_dict[var_a].nbytes)

    def sum_in_variable(self, var, garr, st=None):
        try:
            a = self.sum_kernel
        except AttributeError:
            self.sum_kernel = self.__get_sum_kernel(garr.size, garr.dtype)
        self.sum_kernel.prepared_async_call(
            self.__grid_sum, self.__block_sum, st,
            garr.gpudata,                                          #P
            self.params_dict['conn_data'][var]['delay'].gpudata,   #P
            self.params_dict['cumpre'][var].gpudata,               #P
            self.params_dict['npre'][var].gpudata,                 #P
            self.params_dict['pre'][var].gpudata,                  #P
            self.access_buffers[var].gpudata,                      #P
            self.access_buffers[var].ld,                           #i
            self.access_buffers[var].current,                      #i
            self.access_buffers[var].buffer_length)                #i

    def __get_sum_kernel(self, num_comps, dtype=np.double):
        template = """
        #define NUM_COMPS %(num_comps)d

        __global__ void sum_input(%(type)s* res, int* delay, int* cumpre,
                                  int* npre, int* pre, %(type)s* pre_buffer,
                                  int ld, int current, int buffer_length)
        {
            // must use block size (32, 32, 1)
            int tidx = threadIdx.x;
            int tidy = threadIdx.y;
            int bid = blockIdx.x;

            int comp;

            __shared__ int num_pre[32];
            __shared__ int pre_start[32];
            __shared__ double input[32][33];

            if(tidy == 0)
            {
                comp = bid * 32 + tidx;
                if(comp < NUM_COMPS)
                {
                    num_pre[tidx] = npre[comp];
                }
            } else if(tidy == 1)
            {
                comp = bid * 32 + tidx;
                if(comp < NUM_COMPS)
                {
                    pre_start[tidx] = cumpre[comp];
                }
            }

            input[tidy][tidx] = 0.0;

            __syncthreads();

            comp = bid * 32 + tidy ;
            if(comp < NUM_COMPS){
               int dl;
               int col;
               int n_pre = num_pre[tidy];
               int start = pre_start[tidy];

               for(int i = tidx; i < n_pre; i += 32)
               {
                   dl = delay[start+i];
                   col = current - dl;
                   if(col < 0)
                   {
                     col = buffer_length + col;
                   }

                   input[tidy][tidx] += pre_buffer[col*ld + pre[start+i]];
               }
            }
            __syncthreads();

            if(tidy < 8)
            {
                input[tidx][tidy] += input[tidx][tidy + 8];
                input[tidx][tidy] += input[tidx][tidy + 16];
                input[tidx][tidy] += input[tidx][tidy + 24];
            }

            __syncthreads();

            if(tidy < 4)
            {
                input[tidx][tidy] += input[tidx][tidy + 4];
           }

            __syncthreads();

            if(tidy < 2)
            {
                input[tidx][tidy] += input[tidx][tidy + 2];
            }

            __syncthreads();

            if(tidy == 0)
            {
                input[tidx][0] += input[tidx][1];
                comp = bid*32+tidx;
                if(comp < NUM_COMPS)
                {
                    res[comp] = input[tidx][0];
                }
            }

        }
        //can be improved
        """
        mod = SourceModule(template % {"num_comps": num_comps,
                                       "type": dtype_to_ctype(dtype)},
                           options=self.compile_options)
        func = mod.get_function("sum_input")
        func.prepare('PPPPPPiii')
        self.__block_sum = (32, 32, 1)
        self.__grid_sum = ((num_comps - 1) // 32 + 1, 1)
        return func
