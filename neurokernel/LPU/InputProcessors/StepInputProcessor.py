import numpy as np

from BaseInputProcessor import BaseInputProcessor
class StepInputProcessor(BaseInputProcessor):
    def __init__(self, variable, uids, val, start, stop):
        super(StepInputProcessor, self).__init__([(variable,uids)], mode=0)
        self.val = val
        self.start = start
        self.stop = stop
        self.var = variable
        self.num = len(uids)
        
    def update_input(self):
        self.variables[self.var]['input'] = self.val*np.ones(self.num,\
                                                    self.dtypes[self.var])
            
    def is_input_available(self):
        return (self.LPU_obj.time >= self.start and
                self.LPU_obj.time < self.stop)
        
    def post_run(self):
        pass
    
