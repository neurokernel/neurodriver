
from collections import OrderedDict
import numpy as np
import h5py
from datetime import datetime
from neurokernel.LPU.OutputProcessors.BaseOutputProcessor import BaseOutputProcessor
import pycuda.driver as cuda
import pycuda.gpuarray as garray

class OutputRecorder(BaseOutputProcessor):
    """
    Store specified output in memory rather than writing to file.

    Parameters
    ----------
    var_list: list
    sample_interval: int
    cache_length: int

    Examples
    --------
    import networx as np
    from neurokernel.LPU.InputProcessors.StepInputProcessor import StepInputProcessor
    from neurokernel.LPU.OutputProcessors.OutputRecorder import OutputRecorder

    dt = 1e-6
    dur = 1.0
    steps = int(dur/dt)

    G = nx.MultiDiGraph()
    G.add_node('neuron0', model = 'HodgkinHuxley2',
               g_K = 36.0, g_Na = 120.0, g_L = 0.3,
               E_Na = 50.0, E_K = -77.0, E_L = -54.354)
    input_processor = StepInputProcessor('I', ['neuron0'], 20.0, 0.2, 0.8)
    output_processor = OutputRecorder(
                        [('V', ['neuron0']), ('spike_state', ['neuron0'])],
                        sample_interval = 100) # sample at 1/(dt*100) = 1e4 sampling rate

    lpu = LPU(dt, 'obj', {'graph': G, 'kwargs': {'class_key': 'model'}},
              device = 0, id = 'test', input_processors=[input_processor],
              output_processors = [output_processor],
              manager = False)
    lpu.run(steps = steps)

    neuron0_response = output_processor.get_output(uids = ['neuron0'])
    print(neuron0_response['neuron0'])

    voltage_response = output_processor.get_output(var = 'V')
    print(voltage_response)

    spike_times = output_processor.get_output(var = 'spike_state', uids = 'neuron0')
    print(spike_times['neuron0'])

    output_processor.to_file('test.h5')
    """
    def __init__(self, var_list, sample_interval = 1, cache_length = 1000):
        super(OutputRecorder, self).__init__(var_list, sample_interval)
        self._output = {}
        self._cache = {}
        self.cache_length = 1000

    @property
    def output(self):
        return self._output

    def pre_run(self):
        for var, d in self.variables.items():
            if var == 'spike_state':
                self.output[var] = {'data':
                                        {'time': [],
                                         'index': []},
                                    'uids': d['uids']}
            else:
                self.output[var] = {'data': [],
                                    'uids': d['uids']}
            self._cache[var] = garray.empty((self.cache_length, len(d['uids'])),
                                           d['output'].dtype)
        self.count = 0
        self.scount = 0
        self.stime_shift = 0

    def get_output_array(self, var):
        if var == 'spike_state':
            return int(self._cache[var].gpudata)+\
                    self.scount*self._cache[var].shape[1]*self._cache[var].dtype.itemsize
        else:
            return int(self._cache[var].gpudata)+\
                   self.count*self._cache[var].shape[1]*self._cache[var].dtype.itemsize

    def process_output(self):
        self.count += 1
        if self.count == self.cache_length:
            for var, d in self.variables.items():
                if var != 'spike_state':
                    self.output[var]['data'].append(self._cache[var].get())
            self.count = 0

    def process_spike_output(self):
        self.scount += 1
        if self.scount == self.cache_length:
            var = 'spike_state'
            t, index = np.where(self._cache[var].get()!=0)
            t = t*self.sim_dt
            nspikes = index.size
            if nspikes:
                self.output[var]['data']['time'].append(t+self.stime_shift)
                self.output[var]['data']['index'].append(index.astype(np.int32))
            self.stime_shift += self.cache_length*self.sim_dt
            self.scount = 0

    def post_run(self):
        if self.count > 0:
            for var, d in self.variables.items():
                if var != 'spike_state':
                    self.output[var]['data'].append(self._cache[var].get()[:self.count,:])
        if self.scount > 0:
            var = 'spike_state'
            t, index = np.where(self._cache[var].get()[:self.scount,:]!=0)
            t = t * self.sim_dt
            nspikes = index.size
            if nspikes:
                self.output[var]['data']['time'].append(t+self.stime_shift)
                self.output[var]['data']['index'].append(index.astype(np.int32))
        for var, d in self.variables.items():
            if var != 'spike_state':
                self.output[var]['data'] = np.vstack(self.output[var]['data'])
            else:
                if len(self.output[var]['data']['time']):
                    self.output[var]['data']['time'] = np.concatenate(self.output[var]['data']['time'])
                    self.output[var]['data']['index'] = np.concatenate(self.output[var]['data']['index'])
                else:
                    self.output[var]['data']['time'] = np.zeros(0, np.double)
                    self.output[var]['data']['index'] = np.zeros(0, np.int32)

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

    def to_file(self, filename):
        """
        Write recorded outputs to file with name `filename`,
        compatible with the file written directly by
        `.FileOutputProcessor`

        parameters
        ----------
        filename: str
                  name of the file to store the results
        """
        with h5py.File(filename, 'w') as h5file:
            h5file.create_dataset('metadata', (), 'i')
            h5file['metadata'].attrs['start_time'] = self.start_time
            h5file['metadata'].attrs['sample_interval'] = self.sample_interval
            h5file['metadata'].attrs['dt'] = self.dt
            h5file['metadata'].attrs[
            'DateCreated'] = datetime.now().isoformat()

            for var, d in self.variables.items():
                if var == 'spike_state':
                    h5file.create_dataset(var + '/data/index',
                                          dtype = np.int32,
                                          maxshape = (None,),
                                          data = self.output[var]['data']['index'])
                    h5file.create_dataset(var + '/data/time',
                                          dtype = np.double,
                                          maxshape = (None,),
                                          data = self.output[var]['data']['time'])
                else:
                    h5file.create_dataset(var + '/data',
                                               dtype = self.output[var]['data'].dtype,
                                               maxshape=(None, len(d['uids'])),
                                               data = self.output[var]['data'])
                h5file.create_dataset(var + '/uids',
                                      data = np.array(d['uids'], dtype = 'S'))
