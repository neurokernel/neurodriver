#!/usr/bin/env python
from abc import ABCMeta, abstractmethod, abstractproperty
from future.utils import with_metaclass
from neurokernel.LPU.NDComponents.NDComponent import NDComponent


class BaseMembraneModel(with_metaclass(ABCMeta, NDComponent)):
    # __metaclass__ = ABCMeta

    accesses = ['I']
    updates = ['V']
