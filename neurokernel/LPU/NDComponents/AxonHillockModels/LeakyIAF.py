from BaseAxonHillockModel import BaseAxonHillockModel

import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from neurokernel.LPU.utils.simpleio import *

cuda_src = """
// %(type)s and %(nneu)d must be replaced using Python string foramtting
#define NNEU %(nneu)d

__global__ void leaky_iaf(
    int neu_num,
    %(type)s dt,
    int      *spk,
    %(type)s *V,
    %(type)s *I,
    %(type)s *Vt,
    %(type)s *Vr,
    %(type)s *R,
    %(type)s *C)
{
    int bid = blockIdx.x;
    int nid = bid * NNEU + threadIdx.x;

    %(type)s v,i,r,c;

    if( nid < neu_num ){
        v = V[nid];
        i = I[nid];
        r = R[nid];
        c = C[nid];

        // update v
        %(type)s bh = exp( -dt/r/c );
        v = v*bh + r*i*(1.0-bh);

        // spike detection
        spk[nid] = 0;
        if( v >= Vt[nid] ){
            v = Vr[nid];
            spk[nid] = 1;
        }

        V[nid] = v;
    }
    return;
}
"""

class LeakyIAF(BaseAxonHillockModel):
    def __init__(self, params_dict, access_buffers, dt,
                 debug=False, LPU=None, cuda_verbose=False):
        if cuda_verbose:
            self.compile_options = ['--ptxas-options=-v']
        else:
            self.compile_options = []
            
        self.num_neurons = params_dict['V'].size
        self.params_dict = params_dict
        self.access_buffers = access_buffers
        self.dt = np.double(dt)
        self.steps = 1
        self.debug = debug
        self.LPU = LPU
        self.I = garray.zeros_like(params_dict['V'])
        self.update = self.get_gpu_kernel(params_dict['V'].dtype)

    def pre_run(self, update_pointers):
        cuda.memcpy_dtod(int(update_pointers['V']),
                         self.params_dict['V'],
                         self.params_dict['V'].nbytes)
    
    def run_step(self, update_pointers, st=None):
        self.sum_in_variable('I', self.I)
        self.update.prepared_async_call(
            self.gpu_grid,
            self.gpu_block,
            st,
            self.num_neurons,                               #i
            self.dt,                                        #d
            update_pointers['spike_state'],                 #P
            update_pointers['V'],                           #P
            self.I.gpudata,                                 #P
            self.params_dict['Vt'].gpudata,                 #P
            self.params_dict['Vr'].gpudata,                 #P
            self.params_dict['R'].gpudata,                  #P
            self.params_dict['C'].gpudata)                  #P

    def get_gpu_kernel( self, dtype=np.double):
        self.gpu_block = (128, 1, 1)
        self.gpu_grid = ((self.num_neurons - 1) / self.gpu_block[0] + 1, 1)
        mod = SourceModule(
                cuda_src % {"type": dtype_to_ctype(dtype),
                            "nneu": self.gpu_block[0] },
                options=self.compile_options)
        func = mod.get_function("leaky_iaf")
        func.prepare('idPPPPPPP')
        return func
        

