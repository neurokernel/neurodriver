#!/usr/bin/env python


import warnings
from abc import ABCMeta, abstractmethod, abstractproperty
import os.path
import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from neurokernel.LPU.utils.simpleio import *

class NDComponent(object):
    __metaclass__ = ABCMeta

    accesses = []
    updates = []
    
    @abstractmethod
    def __init__(self, params_dict, access_buffers, dt, debug=False,
                 LPU_id=None, cuda_verbose=False):
        pass
        
    @abstractmethod
    def run_step(self, update_pointers):
        pass


    def pre_run(self, update_pointers):
        pass
        
    def post_run(self):
        '''
        This method will be called at the end of the simulation.
        '''
        pass
    
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
                   dl = delay[i];
                   col = current - dl;
                   if(col < 0)
                   {
                     col = buffer_length + col;
                   }
                                 
                   input[tidy][tidx] += pre_buffer[col*ld + pre[start] + i];
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
        self.__grid_sum = ((num_comps - 1) / 32 + 1, 1)
        return func

