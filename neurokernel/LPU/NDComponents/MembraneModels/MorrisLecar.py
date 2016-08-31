from baseneuron import BaseNeuron

import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

class MorrisLecar(BaseNeuron):
    def __init__(self, params_dict, access_buffers, dt, LPU=None,
                 debug=False, cuda_verbose=False):
        if cuda_verbose:
            self.compile_options = ['--ptxas-options=-v']
        else:
            self.compile_options = []
            
        self.num_neurons = params_dict['V1'].size
            
        self.dt = np.double(dt)
        self.steps = max(int(round(dt / 1e-5)), 1)
        self.debug = debug
        self.ddt = dt / self.steps

        self.LPU = LPU

        # TODO: use LPU object to request memory from memory_manager
        # for even internal states
        self.I = garray.zeros_like(params_dict['initV'])
        self.n = garray.empty_like(params_dict['initn'])
        cuda.memcpy_dtod(self.n, params_dict['initn'],
                         params_dict['initn'].nbytes)

        self.params_dict = params_dict
        self.access_buffers = access_buffers
        self.update = self.get_euler_kernel(params_dict['initV'].dtype)

    def pre_run(self, update_pointers):
        cuda.memcpy_dtod(int(update_pointers['V']),
                         self.params_dict['initV'],
                         self.params_dict['initV'].nbytes)
        

    def run_step(self, update_pointers, st=None):
        self.sum_in_variable('I', self.I)
        self.update.prepared_async_call(
            self.update_grid, self.update_block, st,
            update_pointers['V'],
            self.n.gpudata,
            self.num_neurons,
            self.I.gpudata,
            self.ddt*1000,
            self.steps,
            self.params_dict['V1'].gpudata,
            self.params_dict['V2'].gpudata,
            self.params_dict['V3'].gpudata, 
            self.params_dict['V4'].gpudata,
            self.params_dict['phi'].gpudata,
            self.params_dict['offset'].gpudata)


    def get_euler_kernel(self, dtype=np.double):
        template = """

    #define NVAR 2
    #define NNEU %(nneu)d //NROW * NCOL


    #define V_L (-0.05)
    #define V_Ca 0.1
    #define V_K (-0.07)
    #define g_Ca 1.1
    #define g_K 2.0
    #define g_L 0.5





    __device__ %(type)s compute_n(%(type)s V, %(type)s n, %(type)s V_3, %(type)s V_4, %(type)s Tphi)
    {
        %(type)s n_inf = 0.5 * (1 + tanh((V - V_3) / V_4));
        %(type)s dn = Tphi * cosh(( V - V_3) / (V_4*2)) * (n_inf - n);
        return dn;
    }

    __device__ %(type)s compute_V(%(type)s V, %(type)s n, %(type)s I, %(type)s V_1, %(type)s V_2, %(type)s offset)
    {
        %(type)s m_inf = 0.5 * (1+tanh((V - V_1)/V_2));
        %(type)s dV = (I - g_L * (V - V_L) - g_K * n * (V - V_K) - g_Ca * m_inf * (V - V_Ca) + offset);
        return dV;
    }


    __global__ void
    hhn_euler_multiple(%(type)s* g_V, %(type)s* delay, %(type)s* g_n, int num_neurons, 
                       %(type)s* I_pre, int ld, int current, int buffer_length,
                       %(type)s dt, int nsteps,
                       %(type)s* V_1, %(type)s* V_2, %(type)s* V_3, 
                       %(type)s* V_4, %(type)s* Tphi, %(type)s* offset,
                       int* pre, int* cumpre, int* npre)
    {
        int bid = blockIdx.x;
        int cart_id = bid * NNEU + threadIdx.x;

        %(type)s I, V, n;
        int dl;
        int col; 
        if(cart_id < num_neurons)
        {
            V = g_V[cart_id];
            I = 0;
            for(int i=cumpre[cart_id]; i< cumpre[cart_id]+npre[cart_id]; i++){
              dl = delay[i];
              col = current - dl;
              if(col < 0)
              {
                col = buffer_length + col;
              }
              I += I_pre[col*ld + pre[i]];
            }
            n = g_n[cart_id];

            %(type)s dV, dn;


            for(int i = 0; i < nsteps; ++i)
            {
               dn = compute_n(V, n, V_3[cart_id], V_4[cart_id], Tphi[cart_id]);
               dV = compute_V(V, n, I, V_1[cart_id], V_2[cart_id], offset[cart_id]);

               V += dV * dt;
               n += dn * dt;
            }


            g_V[cart_id] = V;
            g_n[cart_id] = n;
        }

    }
    """ 
        scalartype = dtype.type if dtype.__class__ is np.dtype else dtype
        self.update_block = (128, 1, 1)
        self.update_grid = ((self.num_neurons - 1) / 128 + 1, 1)
        mod = SourceModule(template % {"type": dtype_to_ctype(dtype),
                           "nneu": self.update_block[0]}, 
                           options=self.compile_options)
        func = mod.get_function("hhn_euler_multiple")


        func.prepare('PPiP'+np.dtype(scalartype).char+'iPPPPPP')
        return func
