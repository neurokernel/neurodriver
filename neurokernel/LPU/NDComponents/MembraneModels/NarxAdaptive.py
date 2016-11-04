from BaseMembraneModel import BaseMembraneModel

#import tables

import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from neurokernel.LPU.utils.simpleio import *

cuda_src = """

    #define NNEU %(nneu)d //NROW * NCOL
    #define Y_max 6   // Lag of 6
    #define U_max 7   // Lag of 7
//    #define Bg 3.99  // parameter for saturation  block
#define BG 100 // NARX saturation block - initially 540
#define Ksat 1000    


    __device__ %(type)s sat(%(type)s beta, %(type)s x)
    {
        return (beta+(x*Ksat-beta)/(1+exp(x*Ksat-beta)))/Ksat;
    }
    
    __device__ %(type)s nt(%(type)s K, %(type)s alpha, %(type)s x)
    {
        return K*pow(x,alpha);
    }
    
    __device__ %(type)s lf(%(type)s a, %(type)s b, %(type)s uk, %(type)s uk1, %(type)s yk1)
    {
        return a*(uk+uk1)-b*yk1;
    }



    __global__ void

    narx_adaptive(
    int neu_num,
    %(type)s dt,
    %(type)s *V,
    %(type)s* I_pre,
            %(type)s* Y_1, %(type)s* Y_2, %(type)s* Y_3, 
            %(type)s* Y_4, %(type)s* Y_5, %(type)s* Y_6, 
            %(type)s* U_1, %(type)s* U_2, %(type)s* U_3, 
            %(type)s* U_4, %(type)s* U_5, %(type)s* U_6, %(type)s* U_7, 
            %(type)s* Uo,  %(type)s* Umo, %(type)s* Um1o, %(type)s* Um2o, 
            %(type)s* Um3o, %(type)s* Um4o, %(type)s* Um5o, %(type)s* Um6o, 
            %(type)s* G1, %(type)s* G2, %(type)s* G3, %(type)s* G4, %(type)s* G5, %(type)s* G6,
            int* Cnt_Y, int* Cnt_U, int* Contrast_Gain)    // add Disease Parameter
    {
        int bid = blockIdx.x;
        int nid = (bid * NNEU) + threadIdx.x;
        
        %(type)s v,I,um,um1,um2,um3,um4,um5,um6,uc,g1,g2,g3,g4,g5,g6,fm,fc,NARX_input;
        int  cnt_Y,cnt_U,contrast_gain,ku,ky,a;

        if(nid < neu_num)
        {   

            contrast_gain = Contrast_Gain[nid];
            
            
            %(type)s y[12] ={Y_1[nid],Y_2[nid],Y_3[nid],Y_4[nid],Y_5[nid],Y_6[nid],0,0,0,0,0,0};
            %(type)s u[14] ={U_1[nid],U_2[nid],U_3[nid],U_4[nid],U_5[nid],U_6[nid],U_7[nid],0,0,0,0,0,0,0};
        
            for( a = 0; a < Y_max; a = a + 1 ){y[a+Y_max] = y[a];}
            for( a = 0; a < U_max; a = a + 1 ){u[a+U_max] = u[a];}

            %(type)s    am=0.0012484394506866417,       bm=-0.9975031210986267;
            
            %(type)s    a1=0.00015174687177639374,      b1=-0.9996965062564472,
                        K1=0.0073779,                   al1=-1.69528,   
                        beta1=56.7017;

            %(type)s    a2=0.002073155636233452,        b2=-0.9958536887275331,   
                        K2=0.022266684710357632,        al2=-1.2384563460468387,   
                        beta2=416.19002756957138;            
                        
            %(type)s    a3=0.0003369984766284639,       b3=-0.9993260030467431,   
                        K3=0.0066009688542815071,       al3=-1.1660506692817525,   
                        beta3=2000.0;
 
            %(type)s    a4=0.0012003566315597568,       b4=-0.9975992867368805,                                           
                        K4=0.0000010000931884041385,    al4=-2.2093228258778383,  
                        beta4=6999.9905331388572;

            %(type)s    a5=0.0024937657285116362,       b5=-0.9950124685429768,   
                        K5=0.017438565232583827,        al5=-1.3444700959816109,   
                        beta5=496.14805024208408;
                        
            %(type)s    a6=0.0007212041952871236,       b6=-0.9985575916094257,   
                        K6=0.00053921972167124226,      al6=-1.794266702113672,   
                        beta6=3100.0004971668282;

            %(type)s theta[15] ={ 8.66998468e-01,
                                  6.75079413e-02,
                                 -9.12842610e+01,
                                  2.48898884e-02,
                                  8.17481987e+00,
                                 -8.41121569e-01,
                                 -1.29035677e+00,
                                 -7.79721854e+01,
                                 -9.84690687e-02,
                                  2.35255885e-02,
                                  1.74602660e+00,
                                 -4.31724964e+00,
                                  2.22325254e+02,
                                  1.59757439e+01,
                                  5.59346962e+01 };

            I     = I_pre[nid];
            if(I<0.0000000001)
            {
                I=0.0000000001;
            }
            cnt_Y = Cnt_Y[nid];
            cnt_U = Cnt_U[nid];
             
            ku = cnt_U + U_max;
            ky = cnt_Y + Y_max;           

		 
            v=      theta[0] *y[ky-1]+        
                    theta[1] *y[ky-3]+        
                    theta[2] *u[ku-5]*u[ku-4]+
                    theta[3]                 +
                    theta[4] *u[ku-6]        +
                    theta[5] *u[ku-4]*y[ky-6]+
                    theta[6] *u[ku-7]        +
                    theta[7] *u[ku-7]*u[ku-6]+
                    theta[8] *y[ky-4]        +
                    theta[9] *y[ky-5]        +
                    theta[10]*u[ku-4]*y[ky-5]+
                    theta[11]*u[ku-4]*y[ky-2]+
                    theta[12]*u[ku-7]*u[ku-3]+
                    theta[13]*u[ku-5]        +
                    theta[14]*u[ku-4];


            um =    lf(am,bm,I,Uo[nid],Umo[nid]);
            uc =    I-um;
            um1 =   lf(a1,b1,I,Uo[nid],Um1o[nid]);
            um2 =   lf(a2,b2,I,Uo[nid],Um2o[nid]);
            um3 =   lf(a3,b3,I,Uo[nid],Um3o[nid]);
            um4 =   lf(a4,b4,I,Uo[nid],Um4o[nid]);
            um5 =   lf(a5,b5,I,Uo[nid],Um5o[nid]);
            um6 =   lf(a6,b6,I,Uo[nid],Um6o[nid]);
            
            g1 =    sat(beta1,nt(K1,al1,um1));
            g2 =    sat(beta2,nt(K2,al2,um2));
            g3 =    sat(beta3,nt(K3,al3,um3));
            g4 =    sat(beta4,nt(K4,al4,um4));            
            g5 =    sat(beta5,nt(K5,al5,um5));
            g6 =    sat(beta6,nt(K6,al6,um6));
            
            fm =    g1 + g2 + g3;

            fc =    g4 + g5 + g6;
            
            NARX_input = sat(BG,fm*um+(fc*uc)*contrast_gain) ; // Disease Transform
            V[nid] = v;            

            
            // Update buffers according to counter position            

            Umo[nid] = um;
            Um1o[nid] = um1;    G1[nid] = g1;
            Um2o[nid] = um2;    G2[nid] = g2;
            Um3o[nid] = um3;    G3[nid] = g3;
            Um4o[nid] = um4;    G4[nid] = g4;            
            Um5o[nid] = um5;    G5[nid] = g5;
            Um6o[nid] = um6;    G6[nid] = g6;
            
                   
            
            switch(cnt_Y) { case 0 : Y_1[nid] = v; break; 
                            case 1 : Y_2[nid] = v; break; 
                            case 2 : Y_3[nid] = v; break; 
                            case 3 : Y_4[nid] = v; break; 
                            case 4 : Y_5[nid] = v; break;
                            case 5 : Y_6[nid] = v; break;
                          }   
        	Uo[nid] = I;
            switch(cnt_U) { case 0 : U_1[nid] = NARX_input; break; 
                            case 1 : U_2[nid] = NARX_input; break; 
                            case 2 : U_3[nid] = NARX_input; break; 
                            case 3 : U_4[nid] = NARX_input; break; 
                            case 4 : U_5[nid] = NARX_input; break; 
                            case 5 : U_6[nid] = NARX_input; break;
                            case 6 : U_7[nid] = NARX_input; break;
                          }


            //Increment Y and U counters
            cnt_Y++;
            cnt_U++;
            
            if(cnt_Y >= Y_max){cnt_Y = 0;}
            if(cnt_U >= U_max){cnt_U = 0;}
            
            Cnt_Y[nid] = cnt_Y;
            Cnt_U[nid] = cnt_U;            
        }

    }
    """ 

class NarxAdaptive(BaseMembraneModel):
    def __init__(self, params_dict, access_buffers, dt,
                 debug=False, LPU_id=None, cuda_verbose=True):
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
        self.LPU_id = LPU_id

        self.I = garray.zeros_like(params_dict['V'])

        self.update = self.get_gpu_kernel(params_dict['V'].dtype)

    def pre_run(self, update_pointers):
        cuda.memcpy_dtod(int(update_pointers['V']),
                         self.params_dict['V'].gpudata,
                         self.params_dict['V'].nbytes)
    
    def run_step(self, update_pointers, st=None):
        self.sum_in_variable('I', self.I)
        self.update.prepared_async_call(
            self.gpu_grid,
            self.gpu_block,
            st,
            self.num_neurons,                               #i
            self.dt,                                        #d
            update_pointers['V'],                           #P
            self.I.gpudata,                                 #P
            self.params_dict['Y_1'].gpudata,                #P
            self.params_dict['Y_2'].gpudata,                #P
            self.params_dict['Y_3'].gpudata,                #P
            self.params_dict['Y_4'].gpudata,                #P
            self.params_dict['Y_5'].gpudata,                #P
            self.params_dict['Y_6'].gpudata,                #P
            self.params_dict['U_1'].gpudata,                #P
            self.params_dict['U_2'].gpudata,                #P
            self.params_dict['U_3'].gpudata,                #P
            self.params_dict['U_4'].gpudata,                #P
            self.params_dict['U_5'].gpudata,                #P
            self.params_dict['U_6'].gpudata,                #P
            self.params_dict['U_7'].gpudata,                #P
            self.params_dict['Uo'].gpudata,                #P
            self.params_dict['Umo'].gpudata,                #P
            self.params_dict['Um1o'].gpudata,                #P
            self.params_dict['Um2o'].gpudata,                #P            
            self.params_dict['Um3o'].gpudata,                #P            
            self.params_dict['Um4o'].gpudata,                #P            
            self.params_dict['Um5o'].gpudata,                #P            
            self.params_dict['Um6o'].gpudata,                #P
            self.params_dict['G1'].gpudata,                #P
            self.params_dict['G2'].gpudata,                #P
            self.params_dict['G3'].gpudata,                #P
            self.params_dict['G4'].gpudata,                #P                        
            self.params_dict['G5'].gpudata,                #P
            self.params_dict['G6'].gpudata,                #P                        
            self.params_dict['Cnt_Y'].gpudata,              #P
            self.params_dict['Cnt_U'].gpudata,               #P
            self.params_dict['Contrast_Gain'].gpudata               #P
           )

    def get_gpu_kernel( self, dtype=np.double):
        self.gpu_block = (128, 1, 1)
        self.gpu_grid = ((self.num_neurons - 1) / self.gpu_block[0] + 1, 1)
        mod = SourceModule(
                cuda_src % {"type": dtype_to_ctype(dtype),
                            "nneu": self.gpu_block[0] },
                options=self.compile_options)
        func = mod.get_function("narx_adaptive")
        func.prepare('idPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP')
        return func
        

