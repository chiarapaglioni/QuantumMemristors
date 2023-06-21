from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RYGate
from q_memristor.circuits.Simulator import IBMQSimulator
from q_memristor.numerical.num_memristor import memristor
import numpy as np
import seaborn as sns
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

simulator = IBMQSimulator(backend_string, shots)

# Original parameters of a and b:
# a = np.pi / 4
# b = np.pi / 5
a_states = [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]
b_states = [0, np.pi / 2, np.pi, (3 * np.pi) / 2, 2 * np.pi]

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


def get_purestate(a_val, b_val):
    return [np.cos(a_val), np.sin(a_val) * np.exp(1j * b_val)]


if __name__ == '__main__':

    # Initialize all combinations of pure states
    for a in a_states:
        for b in b_states:
            pure_states.append(get_purestate(a, b))
            labels.append('a: ' + str(a) + ' b: ' + str(b))

    for i in range(len(pure_states)):
        print('Pure State: ', labels[i])

        mem = memristor(y0, w, h, m, a[i], b[i])

        # Single time-step
        t = 0.1

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
        # circuit.draw('mpl', filename='1t_circuit.png')

        counts, measurements, exp_value = simulator.execute_circuit(circuit)

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
    sns.heatmap(V, annot=True, cmap='YlGnBu')

    # Set labels and title
    plt.xlabel('a')
    plt.ylabel('b')
    plt.title('Voltage')

    plt.show()
