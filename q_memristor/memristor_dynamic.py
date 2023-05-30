from qiskit import Aer, execute, QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
from qiskit.circuit.library import RYGate

"""
    Simulation of dynamic simulation quantum memristor 
    (Circuit shown in Fig. 3. of "Quantum Memristors with Quantum Computers")
"""


class IBMQSimulator:
    def __init__(self, backend='qasm_simulator'):
        self.backend = backend

    def execute_circuit(self, circ, shots=1024):
        simulator = Aer.get_backend(self.backend)
        job = execute(circ, simulator, shots=shots)
        result = job.result()
        counts = result.get_counts(circ)
        return counts


def gamma():
    # From Fig. 2 of the paper
    return 0.2


def k(time):
    return 0+time


# # Time-steps
eps = 0.1
tmax = 1
t = np.arange(0, tmax, eps)

# Simulation parameters
# theta = np.arccos(np.exp(k(t)))
theta = np.pi
theta1 = np.pi
phi1 = np.pi
lambda1 = np.pi
# theta2 = np.pi
# phi2 = np.pi
# lambda2 = np.pi

# Initialize registers
Q_env = QuantumRegister(len(t), 'Q_env')
Q_sys = QuantumRegister(1, 'Q_sys')
C = ClassicalRegister(1, 'C')

# Create a quantum circuit with two qubits
circuit = QuantumCircuit(Q_env, Q_sys, C)
print(circuit.draw())

evol_qc = QuantumCircuit(Q_env, Q_sys, name='evolution')
# Implementation of controlled RY gate
cry = RYGate(theta).control(1)
# Apply cry gate to each timestep of the evolution
for i in range(len(t)):
    evol_qc.append(cry, [Q_sys, Q_env[len(t)-1-i]])
    evol_qc.cnot(Q_env[len(t)-1-i], Q_sys)
# print(evol_qc.draw())

all_qubits = Q_env[:] + Q_sys[:]
# print(len(all_qubits))

# Apply gates to circuit
# u = U3 gate
circuit.u(theta1, phi1, lambda1, Q_sys)
circuit.append(evol_qc.to_instruction(), all_qubits)
circuit.u(theta1, phi1, lambda1, Q_sys)

# # Measurement
# # first parameter = 1 = qbit on which the measurement takes place
# # second parameter = 2 = classical bit to place the measurement result in
circuit.measure(Q_sys, C)

print(circuit.draw())
print(circuit.decompose().draw())

# Save image of final circuit
circuit.decompose().draw('mpl', filename='dynamic_circuit.png')

# Unused:
# # Initialize the second qubit in the desired state (1 = index of second qbit)
# # circuit.initialize(desired_state, 1)
