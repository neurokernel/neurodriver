import numpy as np

from .BaseInputProcessor import BaseInputProcessor


class StepInputProcessor(BaseInputProcessor):

    def __init__(self, variable, uids, val, start, stop):
        super(StepInputProcessor, self).__init__([(variable, uids)],
                                                 mode = 1,
                                                 memory_mode = 'gpu')
        self.val = val
        self.start = start
        self.stop = stop
        self.var = variable
        self.num = len(uids)
        self.started = False
        self.stopped = False

    def update_input(self):
        if self.stopped:
            self.variables[self.var]['input'].fill(0)
        else:
            if self.started:
                self.variables[self.var]['input'].fill(self.val)
        # if self.LPU_obj.time == self.start:
        #     self.variables[self.var]['input'].fill(self.val) # * np.ones(self.num, self.dtypes[self.var])
        # else:
        #     self.variables[self.var]['input'].fill(0)

    def is_input_available(self):
        if not self.started:
            if self.LPU_obj.time >= self.start:
                self.started = True
                return True
        else:
            if not self.stopped:
                if self.LPU_obj.time >= self.stop:
                    self.stopped = True
                    return True
        return False

    def post_run(self):
        super(StepInputProcessor, self).post_run()
