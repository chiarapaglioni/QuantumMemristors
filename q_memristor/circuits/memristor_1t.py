from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RYGate

from q_memristor.circuits.csv_generator import csv_gen
from q_memristor.circuits.simulator import IBMQSimulator
from q_memristor.numerical.num_memristor import memristor
import numpy as np
import matplotlib.pyplot as plt


"""
    Circuit simulation of single-time-step quantum memristor based on the article "Quantum Memristors with Quantum 
    Computers" from Y.-M. Guo, F. Albarr ́an-Arriagada, H. Alaeian, E. Solano, and G. Alvarado Barrios. 

    Author: Chiara Paglioni
    Link to Article: https://link.aps.org/doi/10.1103/PhysRevApplied.18.024082 
    
    Experiment Description: 
    The single-time-step quantum circuit was tested for 25 different combinations of coefficients a and b. 
    For each of the combinations the initial values of the voltage and current are calculated after a single 
    evolutionary step. Finally, the circuit is run through the qasm_simulator for 50000 shots.
"""

backend_string = 'qasm_simulator'
shots = 50000

sim = IBMQSimulator(backend_string, shots)

# Original parameters of a and b:
a0 = np.pi / 4
b0 = np.pi / 5
# a_states = [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]
# b_states = [0, np.pi / 2, np.pi, (3 * np.pi) / 2, 2 * np.pi]
a_states = [0, np.pi / 9, np.pi / 6, np.pi / 5, np.pi / 4, np.pi / 3, np.pi / 2]
b_states = [0, np.pi / 6, np.pi / 4, np.pi / 2, np.pi, (3 * np.pi) / 2, 2 * np.pi]

# a_labels = ['0', '\u03C0/6', '\u03C0/4', '\u03C0/3', '\u03C0/2']
# b_labels = ['0', '\u03C0/2', '\u03C0', '3\u03C0/2', '2\u03C0']
a_labels = ['0', '\u03C0/9', '\u03C0/6', '\u03C0/5', '\u03C0/4', '\u03C0/3', '\u03C0/2']
b_labels = ['0', '\u03C0/6', '\u03C0/4', '\u03C0/2', '\u03C0', '3\u03C0/2', '2\u03C0']

# print(a_labels)
# print(b_labels)

y0 = 0.4
w = 1
m = 1
h = 1

pure_states = []
labels = []
exp_values = []
V = []
I = []

zero_state = [1, 0]

# Since the values of a and b are not used in any of the following computation from the memristor class, their initial
# values are chosen as initial parameters
mem = memristor(y0, w, h, m, a0, b0)

# Single time-step
t = 0.1


def get_purestate(a_val, b_val):
    return [np.cos(a_val), np.sin(a_val) * np.exp(1j * b_val)]


if __name__ == '__main__':

    # Initialize all combinations of pure states
    for a in range(len(a_states)):
        for b in range(len(b_states)):
            pure_states.append(get_purestate(a_states[a], b_states[b]))
            labels.append('a: ' + a_labels[a] + ' b: ' + b_labels[b])

    for i in range(len(pure_states)):
        print('Pure State: ', labels[i])

        # Simulation parameters
        theta = np.arccos(np.exp(mem.k1(t)))

        # Initialize registers
        Q_env = QuantumRegister(1, 'Q_env')
        Q_sys = QuantumRegister(1, 'Q_sys')
        C = ClassicalRegister(1, 'C')

        # Create a quantum circuit with two qubits
        circuit = QuantumCircuit(Q_env, Q_sys, C)
        # print(circuit.draw())

        # Implementation of controlled RY gate
        cry = RYGate(theta).control(1)

        # Apply gates to circuit

        # INITIALIZATION
        circuit.initialize(zero_state, Q_env)
        circuit.initialize(pure_states[i], Q_sys)

        # EVOLUTION
        circuit.append(cry, [Q_sys, Q_env])
        circuit.cnot(Q_env, Q_sys)

        # MEASUREMENT
        # Apply gates to perform measurement for Pauli-y
        circuit.sdg(Q_sys)
        circuit.h(Q_sys)

        # first parameter = 1 = qbit on which the measurement takes place
        # second parameter = 2 = classical bit to place the measurement result in
        circuit.measure(Q_sys, C)

        # print(circuit.draw())
        # print(circuit.decompose().draw())

        # Save image of final circuit
        # circuit.draw('mpl', filename='figures/1t_circuit.png')

        counts, measurements, exp_value = sim.execute_circuit(circuit)

        print('Simulator Counts: ', counts)
        # print('Simulator Measurements: ', measurements)
        print('Simulator Expectation Value: ', exp_value)

        # Results for initial values at t = 0.1
        #
        # Simulator Expectation Value:  0.5888
        # Voltage:  -0.2081722363813196
        # Current:  -0.013426173789837254
        v_val = -(1 / 2) * np.sqrt((m * h * w) / 2) * exp_value
        i_val = mem.gamma(t) * v_val

        V.append(v_val)
        I.append(i_val)
        exp_values.append(exp_value)

        print('Voltage: ', v_val)
        print('Current: ', i_val)
        print()

    # Plot voltage and current based on the values of the coefficients a and b
    a_vals = np.array(a_states)
    b_vals = np.array(b_states)

    a_mesh, b_mesh = np.meshgrid(a_vals, b_vals)

    V_arr = np.array(V)
    I_arr = np.array(I)
    exp_values_arr = np.array(exp_values)

    # Save data into csv file
    csvGen = csv_gen('data/data_mem_1t.csv')
    csvGen.write_data(labels, V_arr, I_arr, exp_values_arr)

    # Plot data in heatmap
    V_arr = V_arr.reshape(a_mesh.shape)
    I_arr = I_arr.reshape(a_mesh.shape)
    exp_values_arr = exp_values_arr.reshape(a_mesh.shape)

    fig, axes = plt.subplots(3, 1, figsize=(7, 9))

    axes[0].imshow(V_arr, cmap='YlGnBu', origin='lower')
    axes[0].set_title('Voltage Heatmap')
    axes[0].set_xlabel('a')
    axes[0].set_ylabel('b')
    axes[0].set_xticks(range(len(a_states)))
    axes[0].set_xticklabels(a_labels)
    axes[0].set_yticks(range(len(b_states)))
    axes[0].set_yticklabels(b_labels)
    axes[0].set_aspect('equal')
    axes[0].figure.colorbar(axes[0].images[0], ax=axes[0])

    axes[1].imshow(I_arr, cmap='YlGnBu', origin='lower')
    axes[1].set_title('Current Heatmap')
    axes[1].set_xlabel('a')
    axes[1].set_ylabel('b')
    axes[1].set_xticks(range(len(a_states)))
    axes[1].set_xticklabels(a_labels)
    axes[1].set_yticks(range(len(b_states)))
    axes[1].set_yticklabels(b_labels)
    axes[1].set_aspect('equal')
    axes[1].figure.colorbar(axes[1].images[0], ax=axes[1])

    axes[2].imshow(exp_values_arr, cmap='YlGnBu', origin='lower')
    axes[2].set_title('Expectation Value Heatmap')
    axes[2].set_xlabel('a')
    axes[2].set_ylabel('b')
    axes[2].set_xticks(range(len(a_states)))
    axes[2].set_xticklabels(a_labels)
    axes[2].set_yticks(range(len(b_states)))
    axes[2].set_yticklabels(b_labels)
    axes[2].set_aspect('equal')
    axes[2].figure.colorbar(axes[2].images[0], ax=axes[2])

    plt.tight_layout()
    fig.savefig('figures/heatmap.png')

    for i in range(len(axes)):
        box = axes[i].get_tightbbox(fig.canvas.get_renderer())
        fig.savefig('figures/subplot{}.png'.format(i), bbox_inches=box.transformed(fig.dpi_scale_trans.inverted()))

    plt.show()
