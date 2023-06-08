from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
from qiskit.circuit.library import RYGate
from q_memristor.circuits.Simulator import IBMQSimulator
from q_memristor.numerical.num_memristor import memristor

"""
    Simulation of single-time-step quantum memristor 
    (Circuit shown in Fig. 2. of "Quantum Memristors with Quantum Computers")
"""

if __name__ == '__main__':

    backend_string = 'qasm_simulator'
    shots = 50000

    simulator = IBMQSimulator(backend_string, shots)

    a = np.pi / 4
    b = np.pi / 5
    y0 = 0.4
    w = 1
    m = 1
    h = 1

    pure_state = [np.cos(a), np.sin(a) * np.exp(1j * b)]
    zero_state = [1, 0]

    mem = memristor(y0, w, h, m, a, b)

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
    circuit.initialize(pure_state, Q_sys)

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
    print(circuit.decompose().draw())

    # Save image of final circuit
    circuit.draw('mpl', filename='1t_circuit.png')

    counts, measurements, exp_value = simulator.execute_circuit(circuit)

    print('Simulator Counts: ', counts)
    print('Simulator Measurements: ', measurements)
    print('Simulator Expectation Value: ', exp_value)

    # Results for initial values at t = 0.1
    #
    # Simulator Expectation Value:  0.5888
    # Voltage:  -0.2081722363813196
    # Current:  -0.013426173789837254
    V = -(1 / 2) * np.sqrt((m * h * w) / 2) * exp_value
    I = mem.gamma(t) * V

    print('Voltage: ', V)
    print('Current: ', I)
