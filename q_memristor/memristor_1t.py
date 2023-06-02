from qiskit import Aer, execute, QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
from qiskit.circuit.library import RYGate

"""
    Simulation of single-time-step quantum memristor 
    (Circuit shown in Fig. 2. of "Quantum Memristors with Quantum Computers")
"""


class IBMQSimulator:
    def __init__(self, backend='qasm_simulator'):
        self.backend = backend

    def execute_circuit(self, circ, shots=50000):
        sim = Aer.get_backend(self.backend)
        job = execute(circ, sim, shots=shots)
        result = job.result()
        cnts = result.get_counts(circ)
        return cnts


# Single time-step
t = 0.2

# Simulation parameters
# theta = np.arccos(np.exp(k(t)))
# TODO: determine correct parameters for the simulation
theta = np.pi
theta1 = np.pi
phi1 = np.pi
lambda1 = np.pi
theta2 = np.pi
phi2 = np.pi
lambda2 = np.pi

# Initialize registers
Q_env = QuantumRegister(1, 'Q_env')
Q_sys = QuantumRegister(1, 'Q_sys')
C = ClassicalRegister(1, 'C')

# Create a quantum circuit with two qubits
circuit = QuantumCircuit(Q_env, Q_sys, C)
print(circuit.draw())

# Implementation of controlled RY gate
cry = RYGate(theta).control(1)

# Apply gates to circuit
# u = U3 gate
# cry = controlled RY gate
circuit.u(theta1, phi1, lambda1, Q_sys)
circuit.append(cry, [Q_sys, Q_env])
circuit.cnot(Q_env, Q_sys)
circuit.u(theta1, phi1, lambda1, Q_sys)

# # Measurement
# # first parameter = 1 = qbit on which the measurement takes place
# # second parameter = 2 = classical bit to place the measurement result in
circuit.measure(Q_sys, C)

print(circuit.draw())
print(circuit.decompose().draw())

# Save image of final circuit
circuit.draw('mpl', filename='1t_circuit.png')

# Execute the circuit using the simulator
simulator = IBMQSimulator()
counts = simulator.execute_circuit(circuit)

print('Simulator Measurement: ', counts)
