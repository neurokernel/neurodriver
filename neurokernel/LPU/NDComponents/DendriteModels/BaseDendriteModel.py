#!/usr/bin/env python

from abc import ABCMeta, abstractmethod, abstractproperty
from neurokernel.LPU.NDComponents.NDComponent import NDComponent

class BaseDendriteModel(NDComponent,object):
    __metaclass__ = ABCMeta

    accesses = ['g']
    updates = ['I']
    
    
