import numpy as np
import matplotlib.pyplot as plt

"""
    Basic structure of SNN with sinusoidal input current

    Author: Chiara Paglioni
"""

if __name__ == '__main__':

    # Spiking Neural Network (SNN) Parameters
    num_neurons = 100       # Number of neurons in the network
    simulation_time = 100   # Simulation time (milliseconds)
    dt = 0.1                # Time step (milliseconds)

    # Neuron Model Parameters
    membrane_potential = np.zeros(num_neurons)
    threshold = 1.0                             # Threshold for spike generation
    reset_potential = 0.0                       # Reset potential after spike
    leak_constant = 10.0                        # Leak constant (controls membrane potential decay)

    # Input Current
    input_current = np.random.rand(num_neurons, int(simulation_time/dt))

    print(input_current)

    # eps = 0.0001
    # tmax = 10
    # ts = np.arange(0, tmax, eps)
    #
    # w = np.full((len(ts)), 10)
    # I0 = np.full((len(ts)), 1)
    # I0[:int(len(ts) / 4)] = 0
    # w[:int(len(ts) / 4)] = 0
    #
    # input_current = np.multiply(I0, np.sin(np.multiply(w, ts)))

    spike_times = []
    spike_indices = []

    for t in range(simulation_time):
        # Compute the change in membrane potential for each neuron
        dV = (input_current[:, t] - membrane_potential) / leak_constant

        # Update the membrane potential
        membrane_potential += dV * dt

        # Check for spike generation
        spike_mask = membrane_potential >= threshold

        # Reset the membrane potential for spiking neurons
        membrane_potential[spike_mask] = reset_potential

        # Store the spike times and indices
        spike_times.extend([t] * np.sum(spike_mask))
        spike_indices.extend(np.nonzero(spike_mask)[0].tolist())

    print(spike_times)
    print(spike_indices)

    # Plot the spiking activity
    plt.plot(spike_times, spike_indices, '.k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron index')
    plt.show()
