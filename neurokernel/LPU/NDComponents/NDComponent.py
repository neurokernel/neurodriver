#!/usr/bin/env python


import warnings
from abc import ABCMeta, abstractmethod, abstractproperty
from collections import OrderedDict
import os.path
import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from neurokernel.LPU.utils.simpleio import *

class NDComponent(object):
    """Abstract base Neurodriver component class.

    # Properties
        params_dict: Dict,
        access_buffers: Dict,
        dt: float
        steps:
        debug: bool
        LPU_id: hashable Python object
        update_date: callable
            PyCUDA wrapper to the CUDA kernel. By default, it is the return
            from 'get_update_func'.

    # Class Properties
        max_dt: float
            The upper bound of the acceptable time step value for the model. If
            None,
        params: Dict, mapping between parameters and their default values.
        accesses: List, list of variables to access.
        states: Dict, mapping between state variables and their default values.
        updates: List, list of variabels to update and pupolate.

    # Methods
        run_step:
            .
        pre_run:
            This function is called prior to the simulation. The primary usage
            of this function is to setup initialization for the simulation. For
            example, set initial values for the state variables of the model.
        post_run:
            This function is called at the end of the simulation.
        get_update_func:
            Get the PyCUDA wrapper for the CUDA kernel.


    # Class Methods
    """
    __metaclass__ = ABCMeta

    max_dt = None
    accesses = []
    updates = []
    params = OrderedDict()
    states = OrderedDict()

    def __init__(self, params_dict, access_buffers, dt, debug=False,
                 LPU_id=None, cuda_verbose=False):
        # get the inherited class instead of NDComponent
        cls = type(self)

        self.params_dict = params_dict
        self.access_buffers = access_buffers
        self.LPU_id = LPU_id
        self.debug = debug
        self.compile_options = ['--ptxas-options=-v'] if cuda_verbose else []

        # get number of components
        nums = [v.size for k,v in self.params_dict.items() if k in cls.params]
        nums += [v.size for k,v in self.access_buffers.items() if k in cls.accesses]
        self.num_comps = nums[0]
        assert(all([x == self.num_comps for x in nums]))

        # get dtype from PyCUDA array
        dtypes = [v.dtype.type for k,v in self.params_dict.items() if k in cls.params]
        dtypes += [v.dtype.type for k,v in self.access_buffers.items()]
        # allow only one float type
        floattype = set([np.float32, np.float64]).intersection(dtypes)
        assert(len(floattype) < 2)
        self.floattype = floattype.pop() if len(floattype) else np.float32
        # allow only one int type
        inttype = set([np.int32, np.int64]).intersection(dtypes)
        assert(len(inttype) < 2)
        self.inttype = inttype.pop() if len(inttype) else np.int32
        # make sure no other type appears
        assert(len(set(dtypes) - set([self.floattype, self.inttype])) == 0)

        # setup data type flag for CUDA code
        if self.floattype is np.float64:
            self.compile_options.append('-DUSE_DOUBLE')
        if self.inttype is np.int64:
            self.compile_options.append('-DUSE_LONG_LONG')

        # calculate dt to use during simulation.
        if cls.max_dt is None:
            self.dt = self.floattype(dt)
            self.steps = 1
        else:
            self.steps = int(np.ceil(dt/cls.max_dt))
            self.dt = self.floattype(dt/self.steps)


        self.states = OrderedDict()
        for k,v in cls.states.items():
            dtype = self.floattype if isinstance(v, float) else self.inttype
            self.states[k] = garray.empty(self.num_comps, dtype = dtype)
            self._set_state(k, v)

        self.inputs = OrderedDict()
        for k in self.accesses:
            dtype = self.access_buffers[k].dtype.type
            assert(dtype == self.floattype or dtype == self.inttype)
            self.inputs[k] = garray.empty(self.num_comps, dtype = dtype)

        self.num_garray = len(self.accesses)+len(self.params)+len(self.states) \
            +len(self.updates)
        self.update_func = self.get_update_func()

    def initialize_states(self):
        for k,v in type(self).states.items():
            self._set_state(k, v)

    def _set_state(self, k, v):
        cls = type(self)
        if k in self.params_dict:
            cuda.memcpy_dtod(self.states[k].gpudata,
                             self.params_dict[k].gpudata,
                             self.params_dict[k].nbytes)
        else:
            if isinstance(v, float):
                self.states[k].fill(self.floattype(v))
            else:
                assert(v in cls.states)
                self.states[k].fill(self.floattype(cls.states[v]))

    @abstractmethod
    def run_step(self, update_pointers):
        pass

    def pre_run(self, update_pointers):
        self.initialize_states()

    def post_run(self):
        '''
        This method will be called at the end of the simulation.
        '''
        pass

    @abstractmethod
    def get_update_func(self):
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
        self.__grid_sum = ((num_comps - 1) / 32 + 1, 1)
        return func
