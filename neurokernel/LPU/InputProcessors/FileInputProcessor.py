import h5py
import numpy as np

import pycuda.driver as cuda
import pycuda.gpuarray as garray
from .BaseInputProcessor import BaseInputProcessor
from .ArrayInputProcessor import ArrayInputProcessor


class FileInputProcessor(ArrayInputProcessor):
    """
    Parameters
    ----------
    filename: str
              Name of the h5d file as input.
    mode:     int
              mode in BaseInputProcessor
              0: default, when the file does not cover the entire simulation,
                 inputs default to 0.
              1: when the file does not cover the entire simulation,
                 inputs defaults to the last state.
    cache_length: int
                  length (number of steps) of cache on GPU memory to preload
                  inputs.

    Examples
    --------
    Input h5d must have the following format:
    1. For 'spike_state', there are two options. First,
        '/spike_state/uids': numpy array of dtype 'S',
                             contains the uids of the nodes recorded,
                             the order of this array determines the index below
        '/spike_state/data/time': numpy array of np.float32/np.double,
                                  contains a **monotonically** increasing time stamps
                                  of spikes from all nodes
        '/spike_state/data/index': numpy array of np.int32
                                   contains the index of every entry of spike time
                                   in 'spike_state/data/time'
        The output file generated by FileOutputProcessor complies with this requirement.

       A second option is to have it as the same format as in 2 below, but have
       only float/double value of 0 (no spike at the time step) or 1 (spike)

    2. For other variables, e.g., 'I':
        '/I/uids': numpy array of dtype 'S',
                  contains the uids of the nodes recorded,
                  the order of this array determines which column each node
                  is receives as input in the 'I/data' array.
        '/I/data': numpy array of dtype np.float32/np.double,
                   contains the value of inputs injected to the nodes.
    """
    def __init__(self, filename, file_mode = 'r', mode = 0, cache_length = 1000, preload = False):
        self.filename = filename
        self.cache_length = cache_length
        inputs = {}
        self.file_mode = file_mode
        self.preload = preload
        with h5py.File(self.filename, self.file_mode) as h5file:
            var_list = []
            for var, g in h5file.items():
                if not isinstance(g, h5py.Group):
                    continue
                uids = [a.decode('utf8') for a in g.get('uids')[()].tolist()]
                if var == 'spike_state' and isinstance(g.get('data'), dict):
                    inputs[var] = {'uids': uids,
                                   'data': {'time': g.get('data/time')[:] if preload else [],
                                            'index': g.get('data/index')[:] if preload else []}}
                else:
                    inputs[var] = {'uids': uids,
                                   'data': g.get('data')[:] if preload else []}

        super(FileInputProcessor, self).__init__(inputs = inputs, mode = mode,
                                                 cache_length = cache_length)

    def add_input(self, var, uids, data):
        """
        Add inputs to the input processor

        Parameters
        ----------
        var: str
             the variable to which input will be injected to the components
        uids: list
              the uids of components to inject to
        data: ndarray or dict
              If `variable` is 'spike_state', then data can be in spike event
              format, as a dictionary {'time': np.ndarray, 'index', np.ndarray},\
              specifying the time and index of spikes.
              For all variables, data can be in stepwise input format,
              as a ndarray of shape (T, N)
              that specifies the input to each of the N components
              at every time step for a total of T steps.
              It is expected that if 'spike_state' input is added more than once,
              they must all be of one form, either in spike event format,
              or in stepwise input format.
        """
        if self.preload:
            super(FileInputProcessor, self).add_input(var, uids, data)
        else:
            if self.file_mode == 'r':
                if var not in self.variables:
                    if var == 'spike_state':
                        if isinstance(data, dict):
                            self.spike_state_format = 'event'
                        else:
                            self.spike_state_format = 'stepwise'
                else:
                    raise TypeError('Multiple {} defined in file'.format(var))
            else:
                with h5py.File(self.filename, 'r+') as h5file:
                    if var in h5file:
                        uid_length = h5file[var]['uids'].shape[0]
                        new_uids = np.concatenate((h5file[var]['uids'][:],
                                                   np.array(uids, dtype = 'S')))
                        del h5file[var]['uids']
                        h5file.create_dataset('{}/uids'.format(var),
                                              new_uids)
                        if var == 'spike_state' and isinstance(data, dict):
                            assert self.spike_state_format == 'event', \
                                   'Spike state format was previously set to stepwise in the file, must use the same format.'
                            spike_time = np.concatenate((h5file[var]['data']['time'][:],
                                                         data['time']))
                            index = np.concatenate((h5file[var]['data']['index'][:],
                                                    data['index'] + uid_length))
                            sort_order = np.argsort(spike_time)
                            del h5file[var]['data']
                            h5file.create_dataset('{}/data/index'.format(var),
                                                  maxshape = (None,),
                                                  dtype = np.int32,
                                                  data = index[sort_order])
                            h5file.create_dataset('{}/data/time'.format(var),
                                                  maxshape = (None,),
                                                  dtype = np.float64,
                                                  data = spike_time[sort_order])
                        else:
                            if var == 'spike_state':
                                assert self.spike_state_format == 'stepwise', \
                                       'Spike state format was previously set to event in the file, must use the same format.'
                            new_data = np.hstack((h5file[var]['data'][:], data))
                            del h5file[var]['data']
                            h5file.create_dataset('{}/data'.format(var),
                                                  maxshape = (None, uid_length),
                                                  dtype = np.float64,
                                                  data = new_data)
                    else:
                        uid_length = len(uids)
                        h5file.create_dataset('{}/uids'.format(var),
                                              maxshape = (None,),
                                              data = np.array(uids, dtype = 'S'))
                        if var == 'spike_state' and isinstance(data, dict):
                            self.spike_state_format = 'event'
                            h5file.create_dataset('{}/data/index'.format(var),
                                                  maxshape = (None,),
                                                  dtype = np.int32,
                                                  data = data['index'])
                            h5file.create_dataset('{}/data/time'.format(var),
                                                  maxshape = (None,),
                                                  dtype = np.float64,
                                                  data = data['time'])
                        else:
                            if var == 'spike_state':
                                self.spike_state_format = 'stepwise'
                            h5file.create_dataset('{}/data'.format(var),
                                                  dtype = np.float64,
                                                  maxshape = (None, uid_length),
                                                  data = data)
            self.add_variables([(var, uids)])

    def append_input(self, var, data):
        if preload:
            raise TypeError('FileInputProcessor in preload mode, cannot append input')
        if self.file_mode == 'r':
            raise FileIOError('File mode is set to "r", cannot write to input.')
        else:
            if var not in self.variables:
                raise KeyError('{} not in file, cannot append to. Use add_input instead.'.format(var))
            else:
                with h5py.File(self.filename, 'r+') as h5file:
                    uid_length = h5file[var]['uids'].shape[0]
                    if var == 'spike_state' and isinstance(data, dict):
                        assert self.spike_state_format == 'event', \
                               'Spike state format was previously set to stepwise in the file, must use the same format.'
                        assert data['time'][0] >= h5file[var]['data']['time'][-1], \
                               'Appended spike time must come after the spike time in spike.'
                        assert data['index'].max() < uid_length,\
                               'Spike index mismatch with uids.'
                        assert data['index'].dtype == np.int32, \
                               'index must be np.int32 type'
                        maxshape = h5file[var]['data']['time'].maxshape
                        if maxshape[0] is None:
                            h5file[var]['data']['time'].resize(
                                (h5file[var]['data']['time'].shape[0] + data['time'].shape[0]))
                            h5file[var]['data']['time'][-data['time'].shape[0]:] = data['time']
                        else:
                            spike_time = np.concatenate((h5file[var]['data']['time'][:],
                                                         data['time']))
                            del h5file[var]['data']['time']
                            h5file.create_dataset('{}/data/time'.format(var),
                                                  maxshape = (None,),
                                                  dtype = np.float64,
                                                  data = spike_time)
                        maxshape = h5file[var]['data']['time'].maxshape
                        if maxshape[0] is None:
                            h5file[var]['data']['index'].resize(
                                (h5file[var]['data']['index'].shape[0] + data['index'].shape[0]))
                            h5file[var]['data']['index'][-data['index'].shape[0]:] = data['index']
                        else:
                            index = np.concatenate((h5file[var]['data']['index'][:],
                                                    data['index']))
                            del h5file[var]['data']['index']
                            h5file.create_dataset('{}/data/index'.format(var),
                                                  maxshape = (None,),
                                                  dtype = np.int32,
                                                  data = index)
                    else:
                        if var == 'spike_state':
                            assert self.spike_state_format == 'stepwise', \
                                   'Spike state format was previously set to event in the file, must use the same format.'
                        assert data.shape[1] == uid_length, \
                               'Data dimension mismatch with uids.'
                        assert data.dtype == np.double, 'data must be of np.double'
                        maxshape = h5file[var]['data'].maxshape
                        if maxshape[0] is None:
                            h5file[var]['data'].resize(
                                (h5file[var]['data'].shape[0] + data.shape[0],
                                 uid_length))
                            h5file[var]['data'][-data.shape[0]:,:] = data
                        else:
                            new_data = np.vstack((h5file[var]['data'][:],
                                                  data))
                            del h5file[var]['data']
                            h5file.create_dataset('{}/data'.format(var),
                                                  dtype = np.float64,
                                                  maxshape = (None, uid_length),
                                                  data = new_data)

    def pre_run(self):
        if self.preload:
            super(FileInputProcessor, self).pre_run()
        else:
            self.h5file = h5py.File(self.filename, 'r')
            self.dsets = {}
            self.cache = {}
            self.counts = {}
            self.end_of_var_in_array = {}
            self.block_total = {}
            self.end_of_var = {}
            self.last_read_index = {}
            for var, g in self.h5file.items():
                if not isinstance(g, h5py.Group):
                    continue
                self.block_total[var] = 0
                if var == 'spike_state':
                    if 'index' in g.get('data'):
                        self.dsets[var] = {'index': g.get('data/index')[:] if self.preload else g.get('data/index'),
                                           'time': g.get('data/time')[:] if self.preload else g.get('data/time')}
                        self.spike_state_format = 'event'
                    else:
                        self.dsets[var] = g.get('data')[:] if self.preload else g.get('data')
                        self.spike_state_format = 'stepwise'
                    self.cache[var] = garray.empty((self.cache_length, len(g['uids'])),
                                                   self.variables[var]['input'].dtype)
                else:
                    self.dsets[var] = g.get('data')[:] if self.preload else g.get('data')
                    self.cache[var] = garray.empty((self.cache_length, len(g['uids'])),
                                                   self.variables[var]['input'].dtype)
                self.last_read_index[var] = 0
                self.counts[var] = 0
                self.end_of_var[var] = False
                self.end_of_var_in_array[var] = False

    def update_input(self):
        super(FileInputProcessor, self).update_input()
        if not self.preload:
            if self.end_of_input:
                self.h5file.close()

    def post_run(self):
        if not self.preload:
            if not self.end_of_input:
                self.h5file.close()
