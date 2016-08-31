#!/usr/bin/env python

from abc import ABCMeta, abstractmethod, abstractproperty
from neurokernel.LPU.NDComponents.NDComponent import NDComponent

class BaseAxonHillockModel(NDComponent):
    __metaclass__ = ABCMeta

    accesses = ['I']
    updates = ['spike_state','V']
    
    
