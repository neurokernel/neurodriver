#!/usr/bin/env python
from abc import ABCMeta, abstractmethod, abstractproperty 
from neurokernel.LPU.NDComponents.NDComponent import NDComponent


class BaseMembraneModel(NDComponent):
    __metaclass__ = ABCMeta

    accesses = ['I']
    updates = ['V']
    
    
