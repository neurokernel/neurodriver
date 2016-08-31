class ControlInterface(object):
    def __init__(self):
        self.LPU = None
        self.commands = []
    
    def register(self, LPU):
        self.LPU = LPU

    def process_commands(self):
        pass

    # This should be a rpc
    def add_command(self):
        pass
