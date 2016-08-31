from LPU import LPU
class BaseInputProcessor(Object):
    def __init__(self, LPUobj, obj_ids=None, mode=0):
        assert(isinstance(LPUobj, LPU))
        self._obj_ids = None
        self._LPU_obj = LPUobj
        
    @property
    def obj_ids(self):
        return self._obj_ids
        
    @obj_ids.setter
    def obj_ids(self, value):
        assert(self.LPU_obj.are_ids_valid(value)):
        self._obj_ids = value

    @property
    def LPU_obj(self):
        return self._LPU_obj
        
    def get_input(self):
        # get one dt of input
        # Should this be GPU memory?
        # How to deal with multiple variables and multiple input processors
        pass

    @property
    def max_one_time_read(self):
        return self._max_one_time_read

    # Should be implemented by child class
    def read(self):
        raise NotImplementedError

    # Should be implemented by child class
    def is_input_available(self):
        raise NotImplementedError

    def pre_run(self):
        pass

    def post_run(self):
        pass
