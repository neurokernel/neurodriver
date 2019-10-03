import numpy as np
import h5py
from datetime import datetime
from neurokernel.LPU.OutputProcessors.BaseOutputProcessor import BaseOutputProcessor


class FileOutputProcessor(BaseOutputProcessor):

    def __init__(self, var_list, filename, sample_interval=1):
        self.fname = filename
        super(FileOutputProcessor, self).__init__(var_list, sample_interval)

    def pre_run(self):
        self.h5file = h5py.File(self.fname, 'w')
        self.h5file.create_dataset('metadata', (), 'i')
        self.h5file['metadata'].attrs['start_time'] = self.start_time
        self.h5file['metadata'].attrs['sample_interval'] = self.sample_interval
        self.h5file['metadata'].attrs['dt'] = self.dt
        self.h5file['metadata'].attrs[
            'DateCreated'] = datetime.now().isoformat()

        for var, d in self.variables.items():
            self.h5file.create_dataset(var + '/data', (0, len(d['uids'])),
                                       d['output'].dtype, maxshape=(None, len(d['uids'])))
            self.h5file.create_dataset(var + '/uids', data=np.array(d['uids'], dtype = 'S'))

    def process_output(self):
        for var, d in self.variables.items():
            self.h5file[var + '/data'].resize((self.h5file[var + '/data'].shape[0] + 1,
                                               len(d['uids'])))
            self.h5file[var + '/data'][-1, :] = d['output'].reshape((1, -1))

    def post_run(self):
        self.h5file.close()
