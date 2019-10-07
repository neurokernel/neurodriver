import numpy as np
import h5py
from datetime import datetime
from neurokernel.LPU.OutputProcessors.BaseOutputProcessor import BaseOutputProcessor
import pycuda.driver as cuda
import pycuda.gpuarray as garray


class FileOutputProcessor(BaseOutputProcessor):

    def __init__(self, var_list, filename, sample_interval = 1,
                 cache_length = 1000):
        self.fname = filename
        self.cache_length = 1000
        super(FileOutputProcessor, self).__init__(var_list, sample_interval)

    def pre_run(self):
        self.h5file = h5py.File(self.fname, 'w')
        self.h5file.create_dataset('metadata', (), 'i')
        self.h5file['metadata'].attrs['start_time'] = self.start_time
        self.h5file['metadata'].attrs['sample_interval'] = self.sample_interval
        self.h5file['metadata'].attrs['dt'] = self.dt
        self.h5file['metadata'].attrs[
            'DateCreated'] = datetime.now().isoformat()

        self.cache = {}
        for var, d in self.variables.items():
            self.h5file.create_dataset(var + '/data', (0, len(d['uids'])),
                                       d['output'].dtype, # need to be changed later
                                       maxshape=(None, len(d['uids'])))
            self.h5file.create_dataset(var + '/uids', data=np.array(d['uids'], dtype = 'S'))
            self.cache[var] = garray.empty((self.cache_length, len(d['uids'])),
                                           d['output'].dtype)
        self.count = 0

    def get_output_array(self, var):
        return int(self.cache[var].gpudata)+\
               self.count*self.cache[var].shape[1]*self.cache[var].dtype.itemsize

    def process_output(self):
        # for var, d in self.variables.items():
        #     data = self.get_output_gpu(var)
        #     cuda.memcpy_dtod(
        #         int(self.cache[var].gpudata)+\
        #         self.count*self.cache[var].shape[1]*self.cache[var].dtype.itemsize,
        #         data.gpudata, data.nbytes)
        self.count += 1

        if self.count == self.cache_length:
            for var, d in self.variables.items():
                self.h5file[var + '/data'].resize(
                    (self.h5file[var + '/data'].shape[0] + self.cache_length, len(d['uids'])))
                self.h5file[var + '/data'][-self.cache_length:, :] = self.cache[var].get()
            self.h5file.flush()
            self.count = 0

    def post_run(self):
        if self.count > 0:
            for var, d in self.variables.items():
                self.h5file[var + '/data'].resize(
                    (self.h5file[var + '/data'].shape[0] + self.count, len(d['uids'])))
                self.h5file[var + '/data'][-self.count:, :] = self.cache[var].get()[:self.count,:]
        self.h5file.close()
