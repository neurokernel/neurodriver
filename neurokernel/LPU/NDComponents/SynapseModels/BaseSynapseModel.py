#!/usr/bin/env python

from abc import ABCMeta, abstractmethod, abstractproperty
from neurokernel.LPU.NDComponents.NDComponent import NDComponent

class BaseSynapseModel(NDComponent):
    __metaclass__ = ABCMeta

    accesses = ['V']
    updates = ['g']
    
    
