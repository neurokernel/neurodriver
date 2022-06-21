
import itertools
import pycuda.driver as cuda
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype, context_dependent_memoize
# import pycuda.elementwise as elementwise
from pycuda.compiler import SourceModule

import numpy as np
import networkx as nx
import h5py
from datetime import datetime

from neurokernel.LPU.LPU import LPU

class BaseInputProcessor(object):
    def __init__(self, var_list, mode = 0, memory_mode = 'cpu',
                 input_file = None, input_interval = 1,
                 sensory_file = None, sensory_interval = 1):
        # var_list should be a list of (variable, uids)
        # If no uids is provided, the variable will be ignored
        # Invalid uids will be ignored
        # Derived classes should update self.variables[var]['input']
        # for each variable in update_input method with a ndarray of
        # length len(uids) and the correct dtype
        self.variables = {var:{'uids':uids,'input':None}
                          for var, uids in var_list if uids is not None}
        self.epoch = 0
        self.dest_inds = {}
        self._LPU_obj = None
        # mode = 0 => provide zero input when no input is available
        # mode = 1 => persist with previous input if no input is available
        self.mode = mode
        self.memory_mode = memory_mode
        assert memory_mode in ['cpu', 'gpu'], "Memory mode must be either 'cpu' or 'gpu'"
        self.input_to_be_processed = True
        self.dtypes = {}
        self._d_input = {}
        self.dest_inds = {}
        self.sensory_file = sensory_file
        self.sensory_interval = sensory_interval
        self.input_file = input_file
        self.input_interval = input_interval
        self.metadata = {}
        self._additional_graph = nx.MultiDiGraph()

    def get_new_graph(self):
        return self._additional_graph

    def add_variables(self, var_list):
        for var, uids in var_list:
            if len(uids):
                if var in self.variables:
                    self.variables[var]['uids'].append(uids)
                else:
                    self.variables[var] = {'uids':uids,'input':None}
        self.variables.update({var:{'uids':uids,'input':None}
                                for var, uids in var_list if len(uids)})

    @property
    def LPU_obj(self):
        return self._LPU_obj

    @LPU_obj.setter
    def LPU_obj(self, value):
        assert(isinstance(value, LPU))
        self._LPU_obj = value
        self.dt = self._LPU_obj.dt
        self.sim_dt = self._LPU_obj.dt
        self.memory_manager = self._LPU_obj.memory_manager

    def run_step(self):
        if not self.is_input_available():
            if self.mode == 0:
                self.input_to_be_processed = False
                # for var in self.variables:
                #     self._d_input[var].fill(0)
            elif self.mode == 1:
                self.input_to_be_processed = True
                pass
            else:
                self.input_to_be_processed = False
                # for var in self.variables:
                #     self._d_input[var].fill(0)
                self.LPU_obj.log_info("Invalid mode for Input Processor. " +\
                                      "Defaulting to mode 0(zero input)")
            self.record()
            return

        self.input_to_be_processed = True
        self.update_input()
        if self.memory_mode == 'cpu':
            for var in self.variables:
                self._d_input[var].set(self.variables[var]['input'])
        #TODO, record should be done also when is_input_available = False
        self.record()


    def inject_input(self, var):
        if var not in self.variables: return
        if not self.input_to_be_processed: return
        buff = self.memory_manager.get_buffer(var)
        # dest_mem = garray.GPUArray((1,buff.size),buff.dtype,
        #                            gpudata=int(buff.gpudata)+\
        #                            buff.current*buff.ld*\
        #                            buff.dtype.itemsize)
        dest_mem = int(buff.gpudata)+buff.current*buff.ld*buff.dtype.itemsize
        # self.add_inds(self._d_input[var], dest_mem, self.dest_inds[var])
        self.add_inds(var, self._d_input[var].gpudata,
                      dest_mem)

    def record(self):
        if self.input_file is not None:
            if self.record_count == 0:
                if not self.is_input_available() and self.mode == 0:
                    for var, d in self.variables.items():
                        var_folder = '{}/data'.format(var)
                        if isinstance(var, str):
                            self.input_file_handle[var_folder].resize((self.input_file_handle[var_folder].shape[0]+1,
                                                                        len(d['uids'])))
                            self.input_file_handle[var_folder][-1,:] = np.zeros((1,d['input'].size))
                        elif isinstance(var, tuple):
                            n = len(var)
                            for i, ind_var in enumerate(var):
                                var_subfolder = '{}/{}'.format(var_folder, ind_var)
                                self.input_file_handle[var_subfolder].resize(
                                                (self.input_file_handle[var_subfolder].shape[0]+1,
                                                                            len(d['uids'])))
                                self.input_file_handle[var_subfolder][-1,:] = np.zeros((1,d['input'][i::n].size))
                else:
                    for var, d in self.variables.items():
                        var_folder = '{}/data'.format(var)
                            # if var == 'spike_state':
                            #     self.input_file_handle[var_folder].resize((self.input_file_handle[var_folder].shape[0]+1,
                            #                                                 len(d['uids'])))
                            #     if self.memory_mode == 'cpu':
                            #         self.input_file_handle[var_folder][-1,:] = d['input'].reshape((1,-1))
                            #     elif self.memory_mode == 'gpu':
                            #         self.input_file_handle[var_folder][-1,:] = self._d_input[var].get().reshape((1,-1))
                            # else:
                        if isinstance(var, str):
                            self.input_file_handle[var_folder].resize((self.input_file_handle[var_folder].shape[0]+1,
                                                                        len(d['uids'])))
                            if self.memory_mode == 'cpu':
                                self.input_file_handle[var_folder][-1,:] = d['input'].reshape((1,-1))
                            elif self.memory_mode == 'gpu':
                                self.input_file_handle[var_folder][-1,:] = self._d_input[var].get().reshape((1,-1))
                        elif isinstance(var, tuple):
                            n = len(var)
                            for i, ind_var in enumerate(var):
                                var_subfolder = '{}/{}'.format(var_folder, ind_var)
                                self.input_file_handle[var_subfolder].resize(
                                                (self.input_file_handle[var_subfolder].shape[0]+1,
                                                                            len(d['uids'])))
                                # self.input_file_handle[var_subfolder][-1,:] = d['input'][i::n].reshape((1,-1))
                                if self.memory_mode == 'cpu':
                                    self.input_file_handle[var_subfolder][-1,:] = d['input'][i::n].reshape((1,-1))
                                elif self.memory_mode == 'gpu':
                                    self.input_file_handle[var_subfolder][-1,:] = self._d_input[var].get()[i::n].reshape((1,-1))
            self.record_count = (self.record_count + 1) % self.input_interval

    # Should be implemented by child class
    def update_input(self):
        raise NotImplementedError

    # Should be implemented by child class
    def is_input_available(self):
        raise NotImplementedError

    def _pre_run(self):
        assert(self.LPU_obj)
        assert all([var in self.memory_manager.variables
                    for var in self.variables.keys()]),\
               (list(self.memory_manager.variables), list(self.variables.keys()))
        self.add_inds_func = {}

        if self.input_file is not None:
            self.input_file_handle = h5py.File(self.input_file, 'w')
            self.input_file_handle.create_dataset('metadata',(),'i')
            self.input_file_handle['metadata'].attrs['dt'] = self.dt
            self.input_file_handle['metadata'].attrs['sample_interval'] = self.input_interval
            self.input_file_handle['metadata'].attrs['DateCreated'] = datetime.now().isoformat()
            for k, v in self.metadata.items():
                self.input_file_handle['metadata'].attrs[k] = v

        for var, d in self.variables.items():
            v_dict =  self.memory_manager.variables[var]
            uids = []
            inds = []
            for uid in d['uids']:
                cd = self.LPU_obj.conn_dict[uid]
                assert(var in cd)
                pre = cd[var]['pre'][0]
                inds.append(v_dict['uids'][pre])

            if isinstance(var, str):
                self.dest_inds[var] = garray.to_gpu(np.array(inds,np.int32))
                self.dtypes[var] = v_dict['buffer'].dtype
                self._d_input[var] = garray.zeros(len(d['uids']),self.dtypes[var])
                if self.memory_mode == 'cpu':
                    self.variables[var]['input'] = cuda.pagelocked_zeros(
                                                            len(d['uids']),
                                                            self.dtypes[var])
                elif self.memory_mode == 'gpu':
                    self.variables[var]['input'] = self._d_input[var]
                self.add_inds_func[var] = get_inds_kernel(self.dest_inds[var].dtype,
                                                          v_dict['buffer'].dtype)

            elif isinstance(var, tuple):
                n = len(var)
                new_inds = list(itertools.chain.from_iterable(
                                [[ind*n+i for i in range(n)] for ind in inds]))
                self.dest_inds[var] = garray.to_gpu(np.array(new_inds, np.int32))
                self.dtypes[var] = v_dict['buffer'].dtype
                self._d_input[var] = garray.zeros(len(d['uids'])*n, self.dtypes[var])
                if self.memory_mode == 'cpu':
                    self.variables[var]['input'] = cuda.pagelocked_zeros(
                                                            len(d['uids'])*n,
                                                            self.dtypes[var])
                elif self.memory_mode == 'gpu':
                    self.variables[var]['input'] = self._d_input[var]
                self.add_inds_func[var] = get_inds_kernel(self.dest_inds[var].dtype,
                                                          v_dict['buffer'].dtype)

            else:
                raise TypeError('variable name must either be a str or a tuple of str')


            if self.input_file is not None:
                self.input_file_handle.create_dataset('{}/uids'.format(var),
                                                      data=np.array(d['uids'],
                                                                    dtype = 'S'))
                if isinstance(var, str):
                    self.input_file_handle.create_dataset('{}/data'.format(var), (0,len(d['uids'])),
                                                          d['input'].dtype,
                                                          maxshape=(None,len(d['uids'])))
                elif isinstance(var, tuple):
                    n = len(var)
                    for ind_var in var:
                        self.input_file_handle.create_dataset('{}/data/{}'.format(var, ind_var), (0,len(d['uids'])),
                                                              d['input'].dtype,
                                                              maxshape=(None,len(d['uids'])))
                else:
                    raise TypeError('variable name must either be a str or a tuple of str')
        self.record_count = 0

        self.pre_run()

    def pre_run(self):
        pass

    def post_run(self):
        pass

    # def add_inds(self, src, dest, inds, dest_shift=0):
    #     """
    #     Set `dest[inds[i]+dest_shift] = src[i] for i in range(len(inds))`
    #     """
    #
    #     assert src.dtype == dest.dtype
    #     try:
    #         func = self.add_inds.cache[(inds.dtype, src.dtype)]
    #     except KeyError:
    #         inds_ctype = dtype_to_ctype(inds.dtype)
    #         data_ctype = dtype_to_ctype(src.dtype)
    #         v = ("{data_ctype} *dest, int dest_shift," +\
    #              "{inds_ctype} *inds, {data_ctype} *src").format(\
    #                     data_ctype=data_ctype,inds_ctype=inds_ctype)
    #         func = elementwise.ElementwiseKernel(v,\
    #         "dest[inds[i]+dest_shift] = dest[inds[i]+dest_shift] + src[i]")
    #         self.add_inds.cache[(inds.dtype, src.dtype)] = func
    #     func(dest, int(dest_shift), inds, src, range=slice(0, len(inds), 1) )

    # add_inds.cache = {}

    def add_inds(self, var, src, dest, dest_shift = 0):
        """
        Set `dest[inds[i]+dest_shift] = src[i] for i in range(len(inds))`
        """
        func = self.add_inds_func[var]
        inds = self.dest_inds[var]
        func.prepared_async_call(
            func.grid, func.block, None,
            dest, int(dest_shift), inds.gpudata, src, inds.size)

    def close_files(self):
        if self.input_file is not None:
            try:
                self.input_file_handle.close()
            except:
                pass

    def __del__(self):
        self.close_files()

    def set_metadata(self, metadata):
        """
        metadata: dictionary of metadata
        """
        self.metadata.update(metadata)

    @property
    def record_settings(self):
        return {'sensory_file': self.sensory_file,
                'sensory_interval': self.sensory_interval,
                'input_file': self.input_file,
                'input_interval': self.input_interval}


@context_dependent_memoize
def get_inds_kernel(inds_dtype, src_dtype):
    template = """
__global__ void update(%(data_ctype)s* dest, int dest_shift,
                       %(inds_ctype)s* inds, %(data_ctype)s* src, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    %(inds_ctype)s ind;
    for(int i = tid; i < N; i += total_threads)
    {
        ind = inds[i]+dest_shift;
        dest[ind] += src[i];
    }
}
"""
    mod = SourceModule(template % {"data_ctype": dtype_to_ctype(src_dtype),
                                   "inds_ctype": dtype_to_ctype(inds_dtype)})
    func = mod.get_function("update")
    func.prepare('PiPPi')
    func.block = (128,1,1)
    func.grid = (16 * cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1)
    return func
