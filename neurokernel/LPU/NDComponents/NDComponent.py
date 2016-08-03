#!/usr/bin/env python

"""
Base neuron class used by LPU.
"""

import warnings
from abc import ABCMeta, abstractmethod, abstractproperty
import os.path
import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from neurokernel.LPU.utils.simpleio import *

class NDComponent(object):
    __metaclass__ = ABCMeta

    accesses = []
    updates = []
    
    @abstractmethod
    def __init__(self, params_dict, access_pointers, update_pointers, dt, debug=False, LPU_id=None, cuda_verbose=False):
        '''
        '''
        pass
        
    @abstractmethod
    def run_step(self):
        pass


    def pre_run(self):
        pass
        
    def post_run(self):
        '''
        This method will be called at the end of the simulation.
        '''
        pass
    
