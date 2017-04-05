import numpy as np

from BaseInputProcessor import BaseInputProcessor
class GaussianNoiseInputProcessor(BaseInputProcessor):
    def __init__(self, variable, uids, mean, std, start = -1, stop = -1):
        super(GaussianNoiseInputProcessor, self).__init__([(variable,uids)],
                                                          mode=0)
        self.mean = mean
        self.std = std
        self.start = start
        self.stop = stop
        self.var = variable
        self.num = len(uids)

    def update_input(self):
        self.variables[self.var]['input'] = self.std*\
        np.array(np.random.randn(self.num), dtype = self.dtypes[self.var]) + self.mean

    def is_input_available(self):
        if self.start>-1. and self.stop>self.start:
            return (self.LPU_obj.time >= self.start and
                    self.LPU_obj.time < self.stop)
        else:
            return False

    def post_run(self):
        pass
