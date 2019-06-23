class LeakyIAF(BaseAxonHillockModel):
    updates = ['spike_state', # (bool)
               'V' # Membrane Potential (mV)
              ]
    # accesses are the input variables of the model
    accesses = ['I'] # (\mu A/cm^2 )
    # params are the parameters of the model that needs to be defined
    # during specification of the model
    params = ['resting_potential', # (mV)
              'threshold', # Firing Threshold (mV)
              'reset_potential', # Potential to be reset to after a spike (mV)
              'capacitance', # (\mu F/cm^2)
              'resistance' # (k\Omega cm.^2)
              ]
    # internals are the variables used to store internal states of the model,
    # and are ordered dict whose keys are the variables and value are the initial values.
    internals = OrderedDict([('internalV', 0.0)]) # Membrane Potential (mV)
    
    def step():
        for i in range(n_steps):
            V = V + I * dt 
            if V > threshold:
                V = reset_potential
                spike_state = 1
