from BaseSynapseModel import BaseSynapseModel

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# The following kernel assumes a maximum of one input connection
# per neuron
cuda_src = """
__global__ void alpha_synapse(
    int num,
    %(type)s dt,
    int *spike,
    int ld,
    int current,
    int buffer_length,
    %(type)s *Ar,
    %(type)s *Ad,
    %(type)s *Gmax,
    %(type)s *a0,
    %(type)s *a1,
    %(type)s *a2,
    %(type)s *cond,
    int *Pre,
    int *npre,
    int *cumpre,
    int* delay)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int tot_threads = gridDim.x * blockDim.x;
    %(type)s ar,ad,gmax;
    %(type)s old_a[3];
    %(type)s new_a[3];
    int pre;

    int col;
    for( int i=tid; i<num; i+=tot_threads ){
        // copy data from global memory to register
        if(npre[i]){
            ar = Ar[i];
            ad = Ad[i];
            gmax = Gmax[i];
            old_a[0] = a0[i];
            old_a[1] = a1[i];
            old_a[2] = a2[i];
            // update the alpha function
            new_a[0] = fmax( 0., old_a[0] + dt*old_a[1] );
            new_a[1] = old_a[1] + dt*old_a[2];
            col = current-delay[i];
            if(col < 0)
            {
                col = buffer_length + col;
            }
            pre = col*ld + Pre[cumpre[i]]; 
            if( spike[pre] )
                new_a[1] += ar*ad;
            new_a[2] = -( ar+ad )*old_a[1] - ar*ad*old_a[0];


            // copy data from register to the global memory
            a0[i] = new_a[0];
            a1[i] = new_a[1];
            a2[i] = new_a[2];
            cond[i] = new_a[0]*gmax;
        }
        else{
            cond[i] = 0;
        }
    }
    return;
}
"""
class AlphaSynapse(BaseSynapseModel):
    accesses = ['spike_state']
    
    def __init__( self, params_dict, access_buffers, dt,
                  LPU_id=None, debug=False, cuda_verbose=False):
        if cuda_verbose:
            self.compile_options = ['--ptxas-options=-v']
        else:
            self.compile_options = []

        self.debug = debug
        self.dt = dt
        self.num = params_dict['gmax'].size
        self.LPU_id = LPU_id
        
        self.params_dict = params_dict
        self.access_buffers = access_buffers

        self.a0   = garray.zeros( (self.num,), dtype=np.float64 )
        self.a1   = garray.zeros( (self.num,), dtype=np.float64 )
        self.a2   = garray.zeros( (self.num,), dtype=np.float64 )
        
        self.update = self.get_gpu_kernel(params_dict['gmax'].dtype)

        
    def run_step(self, update_pointers, st = None):
        self.update.prepared_async_call(
            self.gpu_grid,\
            self.gpu_block,\
            st,\
            self.num,\
            self.dt,\
            self.access_buffers['spike_state'].gpudata,
            self.access_buffers['spike_state'].ld,
            self.access_buffers['spike_state'].current,
            self.access_buffers['spike_state'].buffer_length,
            self.params_dict['ar'].gpudata,\
            self.params_dict['ad'].gpudata,\
            self.params_dict['gmax'].gpudata,\
            self.a0.gpudata,\
            self.a1.gpudata,\
            self.a2.gpudata,\
            update_pointers['g'],
            self.params_dict['pre']['spike_state'].gpudata,
            self.params_dict['npre']['spike_state'].gpudata,
            self.params_dict['cumpre']['spike_state'].gpudata,
            self.params_dict['conn_data']['spike_state']['delay'].gpudata)

    def get_gpu_kernel(self, dtype=np.double):
        self.gpu_block = (128,1,1)
        self.gpu_grid = (min( 6*cuda.Context.get_device().MULTIPROCESSOR_COUNT,\
                              (self.num-1)/self.gpu_block[0] + 1), 1)
        mod = SourceModule( \
                cuda_src % {"type": dtype_to_ctype(dtype)},\
                            options=self.compile_options)
        func = mod.get_function("alpha_synapse")
        func.prepare('idPiiiPPPPPPPPPPP')
        return func
