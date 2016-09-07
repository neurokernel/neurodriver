from BaseSynapseModel import BaseSynapseModel

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

#This class assumes a single pre synaptic connection per component instance
class PowerGPotGPot(BaseSynapseModel):
    def __init__(self, params_dict, access_buffers, dt,
                 LPU_id=None, debug=False, cuda_verbose=False):
        if cuda_verbose:
            self.compile_options = ['--ptxas-options=-v']
        else:
            self.compile_options = []

        self.debug = debug
        self.dt = dt
        self.params_dict = params_dict
        self.access_buffers = access_buffers
        self.LPU_id = LPU_id

        self.num_synapse = params_dict['threshold'].size
        self.update_func = self.get_update_func()

    def run_step(self, update_pointers, st = None):
        self.update_func.prepared_async_call(\
                self.grid, self.block, st,
                self.access_buffers['V'].gpudata,                    #P
                self.access_buffers['V'].ld,                         #i
                self.access_buffers['V'].current,                    #i
                self.access_buffers['V'].buffer_length,              #i
                self.params_dict['pre']['V'].gpudata,                #P              
                self.params_dict['npre']['V'].gpudata,               #P               
                self.params_dict['cumpre']['V'].gpudata,             #P                 
                update_pointers['g'],                                #P
                self.params_dict['threshold'].gpudata,               #P               
                self.params_dict['slope'].gpudata,                   #P          
                self.params_dict['power'].gpudata,                   #P          
                self.params_dict['saturation'].gpudata,              #P               
                self.params_dict['conn_data']['V']['delay'].gpudata) #P
                


    def get_update_func(self, dtype=np.double):
        template = """
        #define N_synapse %(n_synapse)d

        __global__ void update(%(type)s* buffer, int buffer_ld, int current,
                               int delay_steps, int* pre, int* npre,
                               int* cumpre, %(type)s* g, %(type)s* thres,
                               %(type)s* slope, %(type)s* power,
                               %(type)s* saturation, int* delay)
        {
            int tid = threadIdx.x + blockIdx.x * blockDim.x;
            int total_threads = gridDim.x * blockDim.x;

            double mem;
            int dl;
            int col;

            for(int i = tid; i < N_synapse; i += total_threads)
            {
                if(npre[i]){
                    dl = delay[i];
                    col = current - dl;
                    if(col < 0)
                    {
                       col = delay_steps + col;
                    }
                    mem = buffer[col*buffer_ld + pre[cumpre[i]]];

                    g[i] = fmin(saturation[i],
                                slope[i]*pow(fmax(0.0,mem-thres[i]),power[i]));
                }
                else{
                    g[i] = 0;
                }
            }

        }
        """
        #Used 14 registers, 64 bytes cmem[0], 4 bytes cmem[16]
        mod = SourceModule(template % {"n_synapse": self.num_synapse,
                                       "type":dtype_to_ctype(dtype)},
                           options=self.compile_options)
        func = mod.get_function("update")
        func.prepare('PiiiPPPPPPPPP')
        #[np.intp, np.int32, np.int32, np.int32, np.intp,
        # np.intp, np.intp, np.intp, np.intp, np.intp, np.intp])
        self.block = (256,1,1)
        self.grid = (min(6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT, (self.num_synapse-1) / 256 + 1), 1)
        return func
