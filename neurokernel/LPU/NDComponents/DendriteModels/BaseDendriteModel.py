#!/usr/bin/env python

from abc import ABCMeta, abstractmethod, abstractproperty
from future.utils import with_metaclass
from neurokernel.LPU.NDComponents.NDComponent import NDComponent

class BaseDendriteModel(with_metaclass(ABCMeta, NDComponent,object)):
    # __metaclass__ = ABCMeta

    accesses = ['g']
    updates = ['I']
