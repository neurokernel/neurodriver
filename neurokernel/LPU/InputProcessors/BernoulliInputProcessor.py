import numpy as np

from .BaseInputProcessor import BaseInputProcessor


class BernoulliInputProcessor(BaseInputProcessor):

    def __init__(self, variable, uids, p, start, stop):
        super(BernoulliInputProcessor, self).__init__([(variable, uids)], mode=0)
        self.p = p
        self.start = start
        self.stop = stop
        self.var = variable
        self.num = len(uids)

    def update_input(self):
        self.variables[self.var]['input'] = np.random.binomial(1, self.p, size=(self.num,)).astype(self.dtypes[self.var])

    def is_input_available(self):
        return (self.LPU_obj.time >= self.start and
                self.LPU_obj.time < self.stop)

    def post_run(self):
        pass
