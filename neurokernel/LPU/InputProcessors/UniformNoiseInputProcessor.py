import numpy as np

from BaseInputProcessor import BaseInputProcessor
class UniformNoiseInputProcessor(BaseInputProcessor):
    def __init__(self, variable, uids, low, high, start = -1, stop = -1):
        super(UniformNoiseInputProcessor, self).__init__([(variable,uids)],
                                                         mode=0)
        self.low = low
        self.high = high
        self.start = start
        self.stop = stop
        self.var = variable
        self.num = len(uids)

    def update_input(self):
        self.variables[self.var]['input'] = \
        np.array(np.random.uniform(low = self.low, high = self.high, size = self.num),\
        dtype = self.dtypes[self.var])

    def is_input_available(self):
        if self.start>-1. and self.stop>self.start:
            return (self.LPU_obj.time >= self.start and
                    self.LPU_obj.time < self.stop)
        else:
            return False

    def post_run(self):
        pass
