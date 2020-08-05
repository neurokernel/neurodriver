

from collections import OrderedDict

import numpy as np
import h5py
from datetime import datetime
from neurokernel.LPU.OutputProcessors.BaseOutputProcessor import BaseOutputProcessor
import pycuda.driver as cuda
import pycuda.gpuarray as garray


class FileOutputProcessor(BaseOutputProcessor):
    def __init__(self, var_list, filename, sample_interval = 1,
                 cache_length = 1000):
        """
        Parameters
        ----------
        var_list: list
                  a list of 2-tuples, where a 2-tuple has the form ('V', uid_list)
                  to indicate the variable 'V' will be recorded for the
                  components with uids in uid_list.
                  If uid_list is None, i.e., ('V', None)
                  all components with update variable 'V' will be recorded.
        filename: str
                  name of the file to store the data
        sample_interval: int
                         Downsample output data by sample_interval and only
                         store the the downsampled. spike_state will always
                         be stored in index/time fashion (see examples)
                         and will not be affected by sample_interval.
        cache_length: int
                      length (number of steps) of cache on GPU memory to store
                      results before they are flushed to storage drive.

        Examples
        --------
        Recorded data can be retrieved by using the following commands:
        1. For 'spike_state':
            with h5py.File('output.h5') as f:
                index=f['spike_state']['data']['index'][:]
                time=f['spike_state']['data']['time'][:]
                uids=f['spike_state']['uids'][:]
            spikes = {uid.decode(): time[index==i] for i, uid in enumerate(uids)}

        2. For other variables, e.g., 'V':
            with h5py.File('output.h5') as f:
                data=f['V']['data'][:]
                uids=f['V']['uids'][:]
                start_time = f['metadata'].attrs['start_time']
                sample_interval = f['metadata'].attrs['sample_interval']
                dt = f['metadata'].attrs['dt']
            t = np.arange(0,data.shape[0]).reshape((-1,1))*dt*sample_interval+start_time
            V = {uid.decode(): np.hstack((t,data[:,i:i+1])) for i, uid in enumerate(uids)}

        3. meta data can be accessed by:
            with h5py.File('output.h5') as f:
                start_time = f['metadata'].attrs['start_time']
                sample_interval = f['metadata'].attrs['sample_interval']
                dt = f['metadata'].attrs['dt']
                DateCreated  = f['metadata'].attrs['DateCreated']
        """
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
            if var == 'spike_state':
                self.h5file.create_dataset(var + '/data/index', (0,), np.int32,
                                           maxshape = (None,))
                self.h5file.create_dataset(var + '/data/time', (0,), np.double,
                                           maxshape = (None,))
                self.h5file.create_dataset(var + '/uids', data=np.array(d['uids'], dtype = 'S'))
                self.cache[var] = garray.empty((self.cache_length, len(d['uids'])),
                                               d['output'].dtype)
            else:
                self.h5file.create_dataset(var + '/data', (0, len(d['uids'])),
                                           d['output'].dtype, # need to be changed later
                                           maxshape=(None, len(d['uids'])))
                self.h5file.create_dataset(var + '/uids', data=np.array(d['uids'], dtype = 'S'))
                self.cache[var] = garray.empty((self.cache_length, len(d['uids'])),
                                               d['output'].dtype)
        self.count = 0
        self.scount = 0
        self.stime_shift = 0

    def get_output_array(self, var):
        if var == 'spike_state':
            return int(self.cache[var].gpudata)+\
                    self.scount*self.cache[var].shape[1]*self.cache[var].dtype.itemsize
        else:
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
                if var != 'spike_state':
                    self.h5file[var + '/data'].resize(
                        (self.h5file[var + '/data'].shape[0] + self.cache_length, len(d['uids'])))
                    self.h5file[var + '/data'][-self.cache_length:, :] = self.cache[var].get()
            self.h5file.flush()
            self.count = 0

    def process_spike_output(self):
        self.scount += 1
        if self.scount == self.cache_length:
            var = 'spike_state'
            t, index = np.where(self.cache[var].get()!=0)
            t = t*self.sim_dt
            nspikes = index.size
            if nspikes:
                self.h5file[var+'/data/time'].resize(
                        (self.h5file[var + '/data/time'].shape[0] + nspikes,))
                self.h5file[var+'/data/index'].resize(
                        (self.h5file[var + '/data/index'].shape[0] + nspikes,))

                self.h5file[var + '/data/time'][-nspikes:] = t+self.stime_shift
                self.h5file[var + '/data/index'][-nspikes:] = index.astype(np.int32)
                self.h5file.flush()
            self.stime_shift += self.cache_length*self.sim_dt
            self.scount = 0

    def post_run(self):
        if self.count > 0:
            for var, d in self.variables.items():
                if var != 'spike_state':
                    self.h5file[var + '/data'].resize(
                        (self.h5file[var + '/data'].shape[0] + self.count, len(d['uids'])))
                    self.h5file[var + '/data'][-self.count:, :] = self.cache[var].get()[:self.count,:]
        if self.scount > 0:
            var = 'spike_state'
            t, index = np.where(self.cache[var].get()[:self.scount,:]!=0)
            t = t * self.sim_dt
            nspikes = index.size
            if nspikes:
                self.h5file[var+'/data/time'].resize(
                        (self.h5file[var + '/data/time'].shape[0] + nspikes,))
                self.h5file[var+'/data/index'].resize(
                        (self.h5file[var + '/data/index'].shape[0] + nspikes,))
                self.h5file[var + '/data/time'][-nspikes:] = t+self.stime_shift
                self.h5file[var + '/data/index'][-nspikes:] = index.astype(np.int32)
        self.h5file.flush()
        self.h5file.close()


class FileOutputReader(object):
    def __init__(self, filename):
        output = {}
        with h5py.File(filename, 'r') as f:
            self.metadata = {'start_time': f['metadata'].attrs['start_time'],
                             'sample_interval': f['metadata'].attrs['sample_interval'],
                             'dt': f['metadata'].attrs['dt'],
                             'DateCreated': f['metadata'].attrs['DateCreated']}
            self.start_time = self.metadata['start_time']
            self.sample_interval = self.metadata['sample_interval']
            self.dt = self.metadata['dt']
            self.DateCreated = self.metadata['DateCreated']

            for var in f:
                if var != 'metadata':
                    if var == 'spike_state':
                        output[var] = {'uids': [n.decode() for n in f[var]['uids'][:]],
                                       'data': {'time': f[var]['data']['time'][:],
                                                'index': f[var]['data']['index'][:]}
                                      }
                    else:
                        output[var] = {'uids': [n.decode() for n in f[var]['uids'][:]],
                                       'data': f[var]['data'][:]}
        self.output = output

    def get_metadata(self):
        return self.metadata

    def get_uids(self, var = None):
        uids = {}
        if var is None:
            for variable in self.output:
                if variable != 'metadata':
                    uids[variable] = self._get_uid_by_var(variable)
        else:
            return self._get_uid_by_var(var)

    def _get_uids_by_var(self, var):
        return self.output[var]['uids']

    def get_output(self, var = None, uids = None):
        """
        retrieve outputs by variable name and uids

        Parameters
        ----------
        var: str
             Name of the variable to retrieve.
             If None, all variables associated with a uid will be retrieved.
        uids: str or list of str
              uids of the component to retrieve
              If not specified (None), all uids associated with the `var`
              will be retrieved.

        Returns
        -------
        output: dict or OrderedDict
                If uids is a list or tuple, returns a OrderedDict, Otherwise a dict,
                with output keyed by uid, and values are either the data/spike_times
                of the component, or a dict keyed by variable name and data/spike_times
                in the value.
        """
        if var is None and uids is None:
            return self.output
        elif var is not None and uids is None:
            return self._get_output_by_var(var)
        elif var is None and uids is not None:
            return self._get_output_by_uids(uids)
        else:
            return self._get_output_by_var_and_uids(var, uids)

    def _get_output_by_var(self, var):
        uids = self.output[var]['uids']
        data = self.output[var]['data']
        if var == 'spike_state':
            output = {uid: {'data': data['time'][data['index']==i] + self.start_time} for i, uid in enumerate(uids)}
        else:
            t = np.arange(0, data.shape[0])*self.dt*self.sample_interval + self.start_time
            output = {uid: {'time': t, 'data': data[:,i].copy()} for i, uid in enumerate(uids)}
        return output

    def _get_output_by_uid(self, uid):
        output = {}
        for var in self.output:
            try:
                index = self.output[var]['uids'].index(uid)
            except ValueError:
                pass
            else:
                if var == 'spike_state':
                    output[var] = {'data': self.output[var]['data']['time'][self.output[var]['data']['index'] == index] + self.start_time}
                else:
                    t = np.arange(0, self.output[var]['data'].shape[0])*self.dt*self.sample_interval + self.start_time
                    output[var] = {'time': t, 'data': self.output[var]['data'][:,index].copy()}
        return output

    def _get_output_by_uids(self, uids):
        if isinstance(uids, str):
            return {uids: self._get_output_by_uid(uids)}
        elif isinstance(uids, (list, tuple)):
            return OrderedDict([(uid, self._get_output_by_uid(uid)) for uid in uids])
        elif isinstance(uids, set):
            return {uid: self._get_output_by_uid(uid) for uid in uids}

    def _get_output_by_var_and_uid(self, var, uid):
        output = None
        try:
            index = self.output[var]['uids'].index(uid)
        except ValueError:
            pass
        else:
            if var == 'spike_state':
                output = {'data': self.output[var]['data']['time'][self.output[var]['data']['index'] == index] + self.start_time}
            else:
                t = np.arange(0, self.output[var]['data'].shape[0])*self.dt*self.sample_interval + self.start_time
                output = {'time': t, 'data': self.output[var]['data'][:,index].copy()}
        return output

    def _get_output_by_var_and_uids(self, var, uids):
        if isinstance(uids, str):
            return {uids: self._get_output_by_var_and_uid(var, uids)}
        elif isinstance(uids, (list, tuple)):
            return OrderedDict([(uid, self._get_output_by_var_and_uid(var, uid)) for uid in uids])
        elif isinstance(uids, set):
            return {uid: self._get_output_by_var_and_uid(var, uid) for uid in uids}
