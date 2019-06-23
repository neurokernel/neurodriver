import pycuda.gpuarray as garray
import numpy as np
import pycuda.elementwise as elementwise
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype

from neurokernel.LPU.LPU import LPU


class BaseInputProcessor(object):
    def __init__(self, var_list, mode=0):
        # var_list should be a list of (variable, uids)
        # If no uids is provided, the variable will be ignored
        # Invalid uids will be ignored
        # Derived classes should update self.variables[var]['input']
        # for each variable in update_input method with a ndarray of
        # length len(uids) and the correct dtype
        self.variables = {var:{'uids':uids,'input':None}
                          for var, uids in var_list if uids}
        self.epoch = 0
        self.dest_inds = {}
        self._LPU_obj = None
        # mode = 0 => provide zero input when no input is available
        # mode = 1 => persist with previous input if no input is available
        self.mode = mode
        self.input_to_be_processed = True
        self.dtypes = {}
        self._d_input = {}
        self.dest_inds = {}

    @property
    def LPU_obj(self):
        return self._LPU_obj

    @LPU_obj.setter
    def LPU_obj(self, value):
        assert(isinstance(value, LPU))
        self._LPU_obj = value
        self.dt = self._LPU_obj.dt
        self.memory_manager = self._LPU_obj.memory_manager

    def run_step(self):
        if not self.is_input_available():
            if self.mode == 0:
                self.input_to_be_processed = False
            elif self.mode == 1:
                self.input_to_be_processed = True
            else:
                self.input_to_be_processed = False
                self.LPU_obj.log_info("Invalid mode for Input Processor. " +\
                                      "Defaulting to mode 0(zero input)")
            return

        self.input_to_be_processed = True
        self.update_input()
        for var in self.variables:
            self._d_input[var].set(self.variables[var]['input'])

    def inject_input(self, var):
        if var not in self.variables: return
        if not self.input_to_be_processed: return
        buff = self.memory_manager.get_buffer(var)
        dest_mem = garray.GPUArray((1,buff.size),buff.dtype,
                                   gpudata=int(buff.gpudata)+\
                                   buff.current*buff.ld*\
                                   buff.dtype.itemsize)
        self.add_inds(self._d_input[var], dest_mem, self.dest_inds[var])

    # Should be implemented by child class
    def update_input(self):
        raise NotImplementedError

    # Should be implemented by child class
    def is_input_available(self):
        raise NotImplementedError

    def _pre_run(self):
        assert(self.LPU_obj)
        assert(all([var in self.memory_manager.variables
                    for var in self.variables.keys()]))
        for var, d in self.variables.items():
            v_dict =  self.memory_manager.variables[var]
            uids = []
            inds = []
            for uid in d['uids']:
                cd = self.LPU_obj.conn_dict[uid]
                assert(var in cd)
                pre = cd[var]['pre'][0]
                inds.append(v_dict['uids'][pre])
            self.dest_inds[var] = garray.to_gpu(np.array(inds,np.int32))
            self.dtypes[var] = v_dict['buffer'].dtype
            self._d_input[var] = garray.zeros(len(d['uids']),self.dtypes[var])
            self.variables[var]['input'] = np.zeros(len(d['uids']),
                                                    self.dtypes[var])
        self.pre_run()

    def pre_run(self):
        pass

    def post_run(self):
        pass

    def add_inds(self, src, dest, inds, dest_shift=0):
        """
        Set `dest[inds[i]+dest_shift] = src[i] for i in range(len(inds))`
        """

        assert src.dtype == dest.dtype
        try:
            func = self.add_inds.cache[(inds.dtype, src.dtype)]
        except KeyError:
            inds_ctype = dtype_to_ctype(inds.dtype)
            data_ctype = dtype_to_ctype(src.dtype)
            v = ("{data_ctype} *dest, int dest_shift," +\
                 "{inds_ctype} *inds, {data_ctype} *src").format(\
                        data_ctype=data_ctype,inds_ctype=inds_ctype)
            func = elementwise.ElementwiseKernel(v,\
            "dest[inds[i]+dest_shift] = dest[inds[i]+dest_shift] + src[i]")
            self.add_inds.cache[(inds.dtype, src.dtype)] = func
        func(dest, int(dest_shift), inds, src, range=slice(0, len(inds), 1) )

    add_inds.cache = {}
