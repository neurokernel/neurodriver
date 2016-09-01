import h5py
import numpy as np

from BaseInputProcessor import BaseInputProcessor
class FileInputProcessor(BaseInputProcessor):
    def __init__(self, filename, mode=0):
        self.h5file = h5py.File(filename, 'r')
        var_list = []
        self.dsets = {}
        for var, g in self.h5file.items():
            if not isintance(g, h5py.Group): continue
            uids = g.get('uids')[()].tolist()
            var_list.append((var, uids))
            self.dsets[var] = g.get('data')
        super(FileInputProcessor, self).__init__(var_list, mode)
            
        self.pointer = 0
        self.end_of_file = False
        
    def update_input(self):
        for var, dset in self.dsets:
            if self.pointer+1 == dset.shape[0]: self.end_of_file=True
            self.variables[var]['input'] = dset[self.pointer,:]
            self.pointer += 1
        if self.end_of_file: self.h5file.close()
            
    def is_input_available(self):
        return self.end_of_file
        
    def post_run(self):
        self.h5file.close()

    
