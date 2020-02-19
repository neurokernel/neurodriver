
from future.utils import iteritems

from .utils import parray
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype, context_dependent_memoize
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
# import pycuda.elementwise as elementwise

import numpy as np
import numbers

class MemoryManager(object):
    def __init__(self,devices=None):
        '''
        TODO : support multiple devices feature. This probably will require
        changes to the neurokernel core as well
        devices should be a list containing the device numbers of the GPUs to be
        used by this MemoryManager
        '''
        self.devices = devices
        self.variables = {}
        self.parameters = {}
        self.mapping = {}          #Mapping from [model_name->variable/parameter]->pos

    def get_buffer(self, variable_name):
        return self.variables[variable_name]['buffer']

    def mutate_variable(self, variable_name, transform):
        pass

    def fill_zeros(self, variable=None, model=None):
        # TODO: Cache dest_inds based on variable and model
        assert(variable or model)
        if variable and not model:
            assert(variable in self.variables)
            d = self.variables[variable]
            dest_inds = np.arange(0, d['cumlen'][-1],1).astype(np.int32)
            buff = d['buffer']
            dest_mem = garray.GPUArray((1,d['cumlen'][-1]),buff.dtype,
                                       gpudata=int(buff.gpudata)+\
                                       buff.current*buff.ld*\
                                       buff.dtype.itemsize)
            dest_mem.fill(0)
        elif model and not variable:
            for var, d in iteritems(self.variables):
                if model in d['models']:
                    mind = d['models'].index(model)
                    stind = d['cumlen'][mind]
                    dest_inds = np.arange(stind, stind+d['len'][mind],1).astype(np.int32)
                    buff = d['buffer']
                    # dest_mem = garray.GPUArray((1,buff.size),buff.dtype,
                    #                            gpudata=int(buff.gpudata)+\
                    #                            buff.current*buff.ld*\
                    #                            buff.dtype.itemsize)
                    dest_mem = int(buff.gpudata)+buff.current*buff.ld*buff.dtype.itemsize
                    self._fill_zeros_kernel(var, model, dest_mem)
        else:
            assert(variable in self.variables)
            d = self.variables[variable]
            if model in d['models']:
                mind = d['models'].index(model)
                stind = d['cumlen'][mind]
                dest_inds = np.arange(stind, stind+d['len'][mind],1).astype(np.int32)
                buff = d['buffer']
                # dest_mem = garray.GPUArray((1,buff.size),buff.dtype,
                #                            gpudata=int(buff.gpudata)+\
                #                            buff.current*buff.ld*\
                #                            buff.dtype.itemsize)
                dest_mem = int(buff.gpudata)+buff.current*buff.ld*buff.dtype.itemsize
                self._fill_zeros_kernel(variable, model, dest_mem)

    def precompile_fill_zeros(self):
        self.fill_zeros_func = {}
        self.dest_inds = {}
        for var, d in iteritems(self.variables):
            self.fill_zeros_func[var] = get_fill_zeros_kernel(d['buffer'].dtype)
            self.dest_inds[var] = {}
            for model in d['models']:
                mind = d['models'].index(model)
                stind = d['cumlen'][mind]
                dest_inds = np.arange(stind, stind+d['len'][mind],1).astype(np.int32)
                self.dest_inds[var][model] = garray.to_gpu(dest_inds)

    def mutate_parameter(self, model_name, param, transform):
        pass

    def memory_alloc(self, variable_name, size, buffer_length=1,
                     dtype=np.double, info={}, init=None):
        assert(variable_name not in self.variables)
        self.variables[variable_name] = {'buffer': \
                            CircularArray(size, buffer_length, dtype, init)}
        self.variables[variable_name].update(info)

    def params_htod(self, model_name, param_dict, dtype=np.double):
        if model_name in self.parameters:
            assert(not (set(self.parameters[model_name]) &
                        set(param_dict)))
        else:
            self.parameters[model_name] = {}

        for k, v in param_dict.items():
            if k in ['pre','npre','cumpre']:
                self.parameters[model_name][k] = \
                                {var: garray.to_gpu(np.array(v[var],np.int32))\
                                 for var in v}
                continue
            if k=='conn_data':
                cd = {}
                for var,data in v.items():
                    cd[var] = {}
                    for d_key,d in data.items():
                        if isinstance(d, list):
                            tmp = np.array(d)
                            if not np.issubdtype(tmp.dtype, np.number):
                                continue
                        elif isinstance(d, dict):
                            continue
                        else:
                            continue
                        if d_key=='delay':
                            cd[var][d_key] = garray.to_gpu(tmp.astype(np.int32))
                        else:
                            cd[var][d_key] = garray.to_gpu(tmp.astype(dtype))
                self.parameters[model_name]['conn_data'] = cd
            if isinstance(v, list):
                tmp = np.array(v)
                if not np.issubdtype(tmp.dtype, np.number):
                    continue
            elif isinstance(v, dict):
                continue
            else:
                continue
            self.parameters[model_name][k] = garray.to_gpu(tmp.astype(dtype))

    def step(self):
        for d in self.variables.values():
            d['buffer'].step()

    def _fill_zeros_kernel(self, var, model, dest):
        func = self.fill_zeros_func[var]
        inds = self.dest_inds[var][model]
        func.prepared_async_call(
            func.grid, func.block, None,
            dest, inds.gpudata, inds.size)

    # def _fill_zeros_kernel(self, dest, inds):
    #     """
    #     Set `dest[inds[i]] = 0 for i in range(len(inds))`
    #     """
    #
    #     try:
    #         func = self._fill_zeros_kernel.cache[(inds.dtype, dest.dtype)]
    #     except KeyError:
    #         inds_ctype = dtype_to_ctype(inds.dtype)
    #         data_ctype = dtype_to_ctype(dest.dtype)
    #         v = ("{data_ctype} *dest," +\
    #              "{inds_ctype} *inds").format(\
    #                     data_ctype=data_ctype,inds_ctype=inds_ctype)
    #         func = elementwise.ElementwiseKernel(v,\
    #         "dest[inds[i]] =0")
    #         self._fill_zeros_kernel.cache[(inds.dtype, dest.dtype)] = func
    #     func(dest, inds, range=slice(0, len(inds), 1) )
    #
    # _fill_zeros_kernel.cache = {}


@context_dependent_memoize
def get_fill_zeros_kernel(data_dtype):
    template = """
__global__ void update(%(data_ctype)s* dest,
                       %(inds_ctype)s* inds,
                       int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    for(int i = tid; i < N; i += total_threads)
    {
        dest[inds[i]] = 0;
    }
}
"""
    mod = SourceModule(template % {"data_ctype": dtype_to_ctype(data_dtype),
                                   "inds_ctype": dtype_to_ctype(np.int32)})
    func = mod.get_function("update")
    func.prepare('PPi')
    func.block = (128,1,1)
    func.grid = (16 * cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1)
    return func


class CircularArray(object):
    """
    Circular buffer to support variables with memory

    Parameters
    ----------
    size : int
        size of the array for each step
    buffer_length : int
        Number of steps into the past to buffer
    dtype : np.dtype
        Data type to be used for the array
    init : dtype
        Initial value for the data. If not specified defaults to zero
    Attributes
    ----------
    size : int
        See above
    buffer_length : int
        See above
    dtype :
        See above
    parr : parray
        Pitched array of dimensions (buffer_length, size)
    current : int
        An integer in [0,buffer_length) representing Current position
        in the buffer.
    Methods
    -------
    step()
        Advance indices of current position in the buffer
    """

    def __init__(self, size, buffer_length, dtype=np.double, init=None):

        self.size = size
        if not isinstance(dtype, np.dtype): dtype = np.dtype(dtype)
        self.dtype = dtype

        self.buffer_length = buffer_length
        if init:
            try:
                init = dtype(init)
                self.parr = parray.ones(
                 (buffer_length, size), dtype) * init
            except:
                self.parr = parray.zeros(
                 (buffer_length, size), dtype)
        else:
            self.parr = parray.zeros(
                 (buffer_length, size), dtype)
        self.current = 0
        self.gpudata = self.parr.gpudata
        self.ld = self.parr.ld


    def step(self):
        """
        Advance indices of current graded potential and spiking neuron values.
        """

        if self.size > 0:
            self.current += 1
            if self.current >= self.buffer_length:
                self.current = 0
