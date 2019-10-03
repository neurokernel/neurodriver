from .LPU import LPU

class Transform(Object):
    def __init__(self, LPUobj):
        assert(isinstance(LPUobj, LPU))
        self._obj_ids = None
        self._LPU_obj = LPUobj
        self._kernel = None
        self._need_to_compile_kernel = True
        self._need_to_generate_kernel = True
        self._variable_name = None
        self._custom_kernel = False
        self._kernel_string = None

    def _compile_kernel(self):
        pass

    def _gen_kernel_code(self):
        pass

    @property
    def compiled_kernel(self):
        if self._need_to_compile_kernel:
            self._compile_kernel()
        return self._kernel

    @property
    def variable_name(self):
        return self._variable_name

    @variable_name.setter
    def variable_name(self, value):
        assert(self.LPU_obj.is_variable(value)):
        self ._variable_name = value

    @property
    def obj_ids(self):
        return self._obj_ids

    @obj_ids.setter
    def obj_ids(self, value):
        assert(self._variable_name)
        self._obj_ids = value

    @property
    def LPU_obj(self):
        return self._LPU_obj

    @property
    def kernel(self):
        if self._need_to_generate_kernel:
            self._gen_kernel_code()
        return self._kernel_string

    @kernel.setter
    def kernel(self, value):
        assert(isinstance(value, string))
        self._custom_kernel = True
        self._need_to_compile_kernel = True
        self._need_to_generate_kernel = False
        self._kernel_string = value
