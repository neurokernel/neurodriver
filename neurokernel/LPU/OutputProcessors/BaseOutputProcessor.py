import pycuda.gpuarray as garray
import numpy as np
from neurokernel.LPU import LPU

class BaseOutputProcessor(object):
    def __init__(self, var_list, sample_interval=1):
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
        
    @property
    def LPU_obj(self):
        return self._LPU_obj

    @LPU_obj.setter
    def LPU_obj(self, value):
        assert(isinstance(LPUobj, LPU))
        self._LPU_obj = value
        self.start_time = self._LPU_obj.time
        self.dt = self._LPU_obj.dt
        self.memory_manager = self._LPU_obj.memory_manager
        
    def run_step(self):
        assert(self.LPU_obj)
        self.epoch += 1
        if self.epoch == sample_interval:
            self.epoch = 0
            for var, d in self.variables.items():
                buff = self.memory_manager.get_buffer(var)
                self.get_inds(buff.gpudata, self._d_output[var],
                              self.src_inds[var], buff.ld*buff.current)
                d['output'] = self._d_output[var].get()
            self.process_output()

    def _pre_run(self):
        assert(self.LPU_obj)
        assert(all([var in self.memory_manager.variables
                    for var in self.variables.keys()]))
        for var, d in self.variables.items():
            v_dict =  self.memory_manager.variables[var]
            if not d['uids']:
                d['uids'] = v_dict['uids']
                self.src_inds[var] = np.arange(len(self.uids))
            else:
                uids = []
                inds = []
                for uid in d['uids']:
                    try:
                        inds.append(v_dict['uids'].index(uid))
                        uids.append(uid)
                    except:
                        pass
                inds = np.array(inds,np.int32)
                o = np.argsort(inds)
                self.src_inds[var] = inds[o]
                d['uids'] = [uids[i] for i in o]
            self._d_output[var] = garray.empty(len(d['uids']),
                                               v_dict['buffer'].dtype)
            d['output']=np.zeros(len(d['uids']), v_dict['buffer'].dtype)
        self.pre_run()

    # Should be implemented by child class
    def pre_run(self):
        pass
        
    # Should be implemented by child class
    def process_output(self):
        pass

    # Should be implemented by child class
    def post_run(self):
        pass

    def get_inds(self, src, dest, inds, src_shift=0):
        """
        Set `dest[i] = src[src_shift+inds[i]] for i in range(len(inds))`
        """

        assert src.dtype == dest.dtype
        try:
            func = self.get_inds.cache[(inds.dtype, src.dtype)]
        except KeyError:
            inds_ctype = dtype_to_ctype(inds.dtype)
            data_ctype = dtype_to_ctype(src.dtype)
            v = ("{data_ctype} *dest, int src_shift " +\
                 "{inds_ctype} *inds, {data_ctype} *src").format(\
                        data_ctype=data_ctype,inds_ctype=inds_ctype)
            func = elementwise.ElementwiseKernel(v,\
                            "dest[i] = src[src_shift+inds[i]]")
            self.get_inds.cache[(inds.dtype, src.dtype)] = func
        func(dest, int(src_shift), inds, src, range=slice(0, len(inds), 1) )

    get_inds.cache = {}
