#define EXP exp%(fletter)s
#define POW pow%(fletter)s
#define ABS fabs%(fletter)s

__global__ void update(int num_comps, %(dt)s dt, int n_steps, 
%(I)s*  g_I, 
%(resting_potential)s*  g_resting_potential, 
%(threshold)s*  g_threshold, 
%(reset_potential)s*  g_reset_potential, 
%(capacitance)s*  g_capacitance, 
%(resistance)s*  g_resistance, 
%(internalV)s*  g_internalV, 
%(spike_state)s*  g_spike_state, 
%(V)s*  g_V){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;
    %(dt)s ddt = dt*1000.; // s to ms
    %(I)s I;
    %(resting_potential)s resting_potential;
    %(threshold)s threshold;
    %(reset_potential)s reset_potential;
    %(capacitance)s capacitance;
    %(resistance)s resistance;
    %(internalV)s internalV;
    %(spike_state)s spike_state;
    %(V)s V;
    for(int i_comp = tid; i_comp < num_comps; i_comp += total_threads) {
    spike = 0;
    I = g_I[i_comp];
    resting_potential = g_resting_potential[i_comp];
    threshold = g_threshold[i_comp];
    reset_potential = g_reset_potential[i_comp];
    capacitance = g_capacitance[i_comp];
    resistance = g_resistance[i_comp];
    internalV = g_internalV[i_comp];
    spike_state = g_spike_state[i_comp];
    V = g_V[i_comp];
    int iter_i;
    for(iter_i = 0; iter_i < n_steps; iter_i++){
        V = ((V) + (((I) * (ddt))));
        if (V>threshold){
            V = reset_potential;
            spike_state = 1;
        }
    }
    
    
    g_resting_potential[i_comp] = resting_potential;
    g_threshold[i_comp] = threshold;
    g_reset_potential[i_comp] = reset_potential;
    g_capacitance[i_comp] = capacitance;
    g_resistance[i_comp] = resistance;
    g_internalV[i_comp] = internalV;
    g_spike_state[i_comp] = spike_state;
    g_V[i_comp] = V;
    }
}