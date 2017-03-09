import numpy as np

from BaseInputProcessor import BaseInputProcessor
class RampInputProcessor(BaseInputProcessor):
    def __init__(self, variable, uids, start_time, duration, start_value, stop_value):
        super(RampInputProcessor, self).__init__([(variable,uids)], mode=0)
        self.duration = duration
        self.start_time = start_time
        self.start_value = start_value
        self.stop_value = stop_value
        self.var = variable
        self.num = len(uids)
        
    def update_input(self):
        current_time = self.LPU_obj.time
        if current_time <= self.start_time:
            val = self.start_value
        elif current_time >= self.start_time + self.duration:
            val = self.stop_value
        else:
            val = (self.stop_value-self.start_value)/self.duration*\
                  (current_time-self.start_time) \
                  + self.start_value
        self.variables[self.var]['input'] = val*np.ones(self.num,\
                                                dtype = self.dtypes[self.var])
            
    def is_input_available(self):
        return True
        
    def post_run(self):
        pass
    
