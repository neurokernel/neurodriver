
from future.utils import listvalues
import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype, context_dependent_memoize
# import pycuda.elementwise as elementwise
from pycuda.compiler import SourceModule
import pycuda.driver as cuda

from neurokernel.LPU.LPU import LPU

class BaseOutputProcessor(object):
    def __init__(self, var_list, sample_interval = 1):
        # var_list should be a list of (variable, uids)
        # Invalid uids will be ignored
        # if uids is None, the entire variable will be outputted
        # If uids are specified, the order will not be respected.
        # Derived classes should access self.variables[variable]['uids']
        # at pre_run to know the order
        # Output will be stored in self.variables[variable]['output']
        # and should be processed by derived classes in process_output
        self.variables = {var:{'uids':uids,'output':None}
                          for var, uids in var_list}
        self.sample_interval = sample_interval
        self.epoch = 0
        self.src_inds = {}
        self._LPU_obj = None
        self._d_output = {}

    @property
    def LPU_obj(self):
        return self._LPU_obj

    @LPU_obj.setter
    def LPU_obj(self, value):
        assert(isinstance(value, LPU))
        self._LPU_obj = value
        self.start_time = self._LPU_obj.time
        self.dt = self._LPU_obj.dt
        self.sim_dt = self._LPU_obj.dt
        self.memory_manager = self._LPU_obj.memory_manager

    def run_step(self):
        assert(self.LPU_obj)
        self.epoch += 1
        if self.epoch == self.sample_interval:
            self.epoch = 0
            for var, d in self.variables.items():
                buff = self.memory_manager.get_buffer(var)
                # src_mem = garray.GPUArray((1,buff.size),buff.dtype,
                #                           gpudata=int(buff.gpudata)+\
                #                           buff.current*buff.ld*\
                #                           buff.dtype.itemsize)
                src_mem = int(buff.gpudata)+buff.current*buff.ld*buff.dtype.itemsize
                dest_mem = self.get_output_array(var)
                if dest_mem is None:
                    dest_mem = self._d_output[var].gpudata
                self.get_inds(var, src_mem, dest_mem)
            self.process_output()

        if 'spike_state' in self.variables:
            var = 'spike_state'
            buff = self.memory_manager.get_buffer(var)
            src_mem = int(buff.gpudata)+buff.current*buff.ld*buff.dtype.itemsize
            dest_mem = self.get_output_array(var)
            if dest_mem is None:
                dest_mem = self._d_output[var].gpudata
            self.get_inds(var, src_mem, dest_mem)
            self.process_spike_output()

    def get_output_array(self, var):
        return None

    def _pre_run(self):
        assert(self.LPU_obj)
        assert(all([var in self.memory_manager.variables
                    for var in self.variables]))
        self.get_inds_func = {}
        for var, d in self.variables.items():
            v_dict =  self.memory_manager.variables[var]
            if not d['uids']:
                uids = list(v_dict['uids'])
                inds = listvalues(v_dict['uids'])
                o = np.argsort(inds)
                d['uids'] = [uids[i] for i in o]
                self.src_inds[var] = garray.to_gpu(np.arange(len(d['uids'])))
            else:
                uids = []
                inds = []
                for uid in d['uids']:
                    try:
                        inds.append(v_dict['uids'][uid])
                        uids.append(uid)
                    except:
                        pass
                inds = np.array(inds,np.int32)
                o = np.argsort(inds)
                self.src_inds[var] = garray.to_gpu(inds[o])
                d['uids'] = [uids[i] for i in o]
            self._d_output[var] = garray.empty(len(d['uids']),
                                               v_dict['buffer'].dtype)
            d['output']=np.zeros(len(d['uids']), v_dict['buffer'].dtype)
            self.get_inds_func[var] = get_inds_kernel(self.src_inds[var].dtype,
                                                      v_dict['buffer'].dtype)
        self.pre_run()

    def get_output_gpu(self, var):
        return self._d_output[var]

    def get_output(self, var):
        self._d_output[var].get(self.variables[var]['output'])
        return self.variables[var]['output']

    # Should be implemented by child class
    def pre_run(self):
        pass

    # Should be implemented by child class
    def process_output(self):
        pass

    def process_spike_output(self):
        pass

    # Should be implemented by child class
    def post_run(self):
        pass

    def get_inds(self, var, src, dest, src_shift = 0):
        func = self.get_inds_func[var]
        inds = self.src_inds[var]
        func.prepared_async_call(
            func.grid, func.block, None,
            dest, int(src_shift), inds.gpudata, src, inds.size)

#     def get_inds1(self, src_dtype, src, dest, inds, src_shift = 0):
#         # assert src.dtype == dest.dtype
#         inds_ctype = dtype_to_ctype(inds.dtype)
#         data_ctype = dtype_to_ctype(src_dtype)
#
#         func = get_inds_kernel1(inds_ctype, data_ctype)
#         func.prepared_async_call(
#             func.grid, func.block, None,
#             dest, int(src_shift), inds.gpudata, src, inds.size)
#
#     def get_inds(self, src, dest, inds, src_shift=0):
#         """
#         Set `dest[i] = src[src_shift+inds[i]] for i in range(len(inds))`
#         """
#
#         assert src.dtype == dest.dtype
#         inds_ctype = dtype_to_ctype(inds.dtype)
#         data_ctype = dtype_to_ctype(src.dtype)
#
#         func = get_inds_kernel(inds_ctype, data_ctype)
#         func(dest, int(src_shift), inds, src, range=slice(0, len(inds), 1) )
#
# @context_dependent_memoize
# def get_inds_kernel(inds_ctype, src_ctype):
#     v = ("{data_ctype} *dest, int src_shift, " +\
#          "{inds_ctype} *inds, {data_ctype} *src").format(\
#                 data_ctype=src_ctype,inds_ctype=inds_ctype)
#     func = elementwise.ElementwiseKernel(v,\
#                     "dest[i] = src[src_shift+inds[i]]")
#     return func


@context_dependent_memoize
def get_inds_kernel(inds_dtype, src_dtype):
    template = """
__global__ void update(%(data_ctype)s* dest, int src_shift,
                       %(inds_ctype)s* inds, %(data_ctype)s* src, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    for(int i = tid; i < N; i += total_threads)
    {
        dest[i] = src[src_shift+inds[i]];
    }
}
"""
    mod = SourceModule(template % {"data_ctype": dtype_to_ctype(src_dtype),
                                   "inds_ctype": dtype_to_ctype(inds_dtype)})
    func = mod.get_function("update")
    func.prepare('PiPPi')
    func.block = (128,1,1)
    func.grid = (16 * cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1)
    return func
