import utils.parray as parray
import numpy as np
import pycuda.gpuarray as garray
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
            assert(not (set(self.parameters[model_name].keys()) &
                        set(param_dict.keys())))
        else:
            self.parameters[model_name] = {}

        for k, v in param_dict.items():
            if k in ['pre','npre','cumpre']:
                self.parameters[model_name][k] = \
                                {var: garray.to_gpu(np.array(v[var],np.int32))\
                                 for var in v.keys()}
                continue
            if k=='conn_data':
                cd = {}
                for var,data in v.items():
                    cd[var] = {}
                    for d_key,d in data.items():
                        if not all([isinstance(i,numbers.Number) for i in d]):
                            continue
                        if d_key=='delay':
                            cd[var][d_key] = garray.to_gpu(np.array(d, np.int32))
                        else:
                            cd[var][d_key] = garray.to_gpu(np.array(d, dtype))
                self.parameters[model_name]['conn_data'] = cd
            if not all([isinstance(i,numbers.Number) for i in v]): continue
            self.parameters[model_name][k] = garray.to_gpu(np.array(v, dtype))
            
    def step(self):
        for d in self.variables.values():
            d['buffer'].step()
            
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
