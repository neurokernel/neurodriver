import numpy as np
import h5py
from datetime import datetime
from neurokernel.LPU.OutputProcessors.BaseOutputProcessor import BaseOutputProcessor
import pycuda.driver as cuda
import pycuda.gpuarray as garray

class OutputRecorder(BaseOutputProcessor):
    """
    Store specified output in memory rather than writing to file.
    """
    def __init__(self, var_list, dur, dt, sample_interval = 1):
        super(OutputRecorder, self).__init__(var_list, sample_interval)
        self.total_steps = int(dur/dt/sample_interval)

    def pre_run(self):
        self.output = {}
        for var, d in self.variables.items():
            self.output[var] = {'data': garray.empty((self.total_steps,
                                                      len(d['uids'])),
                                                     dtype = self._d_output[var].dtype),
                      'uids': np.array(d['uids'], dtype = 'S')}
        self.count = 0

    def get_output_array(self, var):
        return int(self.output[var]['data'].gpudata)+\
               self.count*self.output[var]['data'].shape[1]*self.output[var]['data'].dtype.itemsize

    def process_output(self):
        self.count += 1

    def post_run(self):
        for var, d in self.variables.items():
            self.output[var]['data'] = self.output[var]['data'].get()
