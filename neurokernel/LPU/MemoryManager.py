import utils.parray as parray


class MemoryManager(object):

    def __init__(self,devices):
    '''
    devices should be a list containing the device numbers of the GPUs to be
    used by this MemoryManager
    '''
        self.devices = devices
        self.variables = {}
    
    def get_memory(self,variable_name):
        pass

    def mutate_memory(self, variable_name, transform):
        pass

    def memory_alloc(self, size, buffer_length=1):
        pass

    

    
    




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
    buffer : parray
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
        self.dtype = dtype
        self.buffer_length = buffer_length
        if init:
            try:
                init = dtype(init)
                self.buffer = parray.ones(
                 (buffer_length, size), dtype) * init
            except:
                self.buffer = parray.zeros(
                 (buffer_length, size), dtype)
        else:
            self.buffer = parray.zeros(
                 (buffer_length, size), dtype)
        self.current = 0

    def step(self):
        """
        Advance indices of current graded potential and spiking neuron values.
        """

        if self.size > 0:
            self.current += 1
            if self.current >= self.buffer_length:
                self.current = 0
