
import time

import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype, context_dependent_memoize
from pycuda.compiler import SourceModule

from .BaseInputProcessor import BaseInputProcessor
from ..utils import curand

class BernoulliInputProcessor(BaseInputProcessor):

    def __init__(self, variable, uids, p, start, stop, seed = None):
        super(BernoulliInputProcessor, self).__init__([(variable, uids)],
                                                      mode = 0,
                                                      memory_mode = 'gpu')
        self.p = p
        self.start = start
        self.stop = stop
        self.var = variable
        self.seed = seed
        self.num = len(uids)

    def update_input(self):
        #self.variables[self.var]['input'] = np.random.binomial(1, self.p, size=(self.num,)).astype(self.dtypes[self.var])
        self.update_kernel.prepared_async_call(
            self.update_kernel.grid, self.update_kernel.block, None,
            self.variables[self.var]['input'].gpudata, self.num,
            self.p, self.state.gpudata)

    def is_input_available(self):
        return (self.LPU_obj.time >= self.start and
                self.LPU_obj.time < self.stop)

    def pre_run(self):
        self.update_kernel = get_update_func(self.dtypes[self.var])
        self.state = curand.curand_setup(
            self.update_kernel.block[0]*self.update_kernel.grid[0],
            self.seed if self.seed is not None else time.monotonic_ns())

    def post_run(self):
        pass


@context_dependent_memoize
def get_update_func(dtype):
    template = """
#include "curand_kernel.h"
extern "C" {
__global__ void draw(%(type)s* output, int N, double p, curandStateXORWOW_t* state)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    curandStateXORWOW_t local_state = state[tid];
    double n;
    for(int i = tid; i < N; i += total_threads)
    {
        n = curand_uniform_double(&local_state);
        output[i] = (n < p) ? 1 : 0;
    }
    state[tid] = local_state;
}
}
"""
    mod = SourceModule(template % {"type": dtype_to_ctype(dtype)},
                       no_extern_c = True)
    func = mod.get_function("draw")
    func.prepare('PidP')
    func.block = (128,1,1)
    func.grid = (16 * cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1)
    return func
