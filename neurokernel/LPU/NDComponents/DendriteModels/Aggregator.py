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
from .BaseDendriteModel import BaseDendriteModel

class Aggregator(BaseDendriteModel):
    accesses = ['g','V']
    updates = ['I']

    def __init__(self, params_dict, access_buffers, dt, debug=False,
                 LPU_id=None, cuda_verbose=False):
        if cuda_verbose:
            self.compile_options = ['--ptxas-options=-v']
        else:
            self.compile_options = []

        self.LPU_id = LPU_id
        self.access_buffers = access_buffers
        self.params_dict = params_dict
        self.debug = debug
        self.dt = dt
        self.LPU_id = LPU_id

        self.num_comps = params_dict['pre']['V'].size


        self.update = self.get_update_func(self.access_buffers['g'].dtype)

    @property
    def maximum_dt_allowed(self):
        return self.dt

    def run_step(self, update_pointers, st=None):
        self.update.prepared_async_call(\
                        self.grid, self.block, st,
                        self.access_buffers['g'].gpudata,                     #P
                        self.access_buffers['g'].ld,                          #i
                        self.access_buffers['g'].current,                     #i
                        self.access_buffers['g'].buffer_length,               #i
                        self.params_dict['conn_data']['g']['delay'].gpudata,  #P
                        self.params_dict['conn_data']['g']['reverse'].gpudata,#P
                        self.params_dict['pre']['g'].gpudata,                 #P
                        self.params_dict['npre']['g'].gpudata,                #P
                        self.params_dict['cumpre']['g'].gpudata,              #P
                        self.access_buffers['V'].gpudata,                     #P
                        self.access_buffers['V'].ld,                          #i
                        self.access_buffers['V'].current,                     #i
                        self.params_dict['pre']['V'].gpudata,                 #P
                        update_pointers['I'])                                 #P


    def get_update_func(self, dtype=np.double):
        template = """
        #define NUM_COMPS %(num_comps)d

        __global__ void aggregate_I(%(type)s* g, int ld, int current,
                                    int buffer_length, int* delay,
                                    %(type)s* V_rev, int* pre, int* npre,
                                    int* cumpre, %(type)s* V, int V_ld,
                                    int V_current, int* V_pre, %(type)s* I)
        {
            // must use block size (32, 32, 1)
            int tidx = threadIdx.x;
            int tidy = threadIdx.y;
            int bid = blockIdx.x;
            int comp;

            __shared__ int num_pre[32];
            __shared__ int pre_start[32];
            __shared__ double V_in[32];
            __shared__ double input[32][33];

            if(tidy == 0)
            {
                comp = bid * 32 + tidx;
                if(comp < NUM_COMPS)
                {
                    num_pre[tidx] = npre[comp];
                    V_in[tidx] = V[V_pre[comp]+V_current*V_ld];
                }
            } else if(tidy == 1)
            {
                comp = bid * 32 + tidx;
                if(comp < NUM_COMPS)
                {
                    pre_start[tidx] = cumpre[comp];
                    I[comp] = 0;
                }
            }

            input[tidy][tidx] = 0.0;

            __syncthreads();

            comp = bid * 32 + tidy ;
            if(comp < NUM_COMPS)
            {
               int dl;
               int col;
               int n_pre = num_pre[tidy];
               int start = pre_start[tidy];
               double VV = V_in[tidy];


               for(int i = tidx; i < n_pre; i += 32)
               {
                   dl = delay[start+i];
                   col = current - dl;
                   if(col < 0)
                   {
                     col = buffer_length + col;
                   }

                   input[tidy][tidx] += g[pre[start + i] + col*ld] *
                                        (VV - V_rev[start + i]);
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
                    I[comp] -= input[tidx][0];
                }
            }
        }
        """
        mod = SourceModule(template % {"num_comps": self.num_comps,
                                       "type": dtype_to_ctype(dtype)},
                           options=self.compile_options)
        func = mod.get_function("aggregate_I")
        func.prepare('PiiiPPPPPPiiPP')
        self.block = (32, 32, 1)
        self.grid = ((self.num_comps - 1) // 32 + 1, 1)
        return func
