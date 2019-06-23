
for i in range(n_steps):
    V = (V + (I * dt))
    if (V > threshold):
        V = reset_potential
        spike_state = 1
