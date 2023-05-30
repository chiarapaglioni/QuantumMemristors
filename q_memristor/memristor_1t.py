from qiskit import Aer, execute, QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
# following package could not be downloaded/imported :(
# from ibm_quantum_widgets import draw_circuit

"""
    Simulation of single-time-step quantum memristor 
    (Circuit shown in Fig. 2. of "Quantum Memristors with Quantum Computers")
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


# In this case gamma is fixed as there is one timestep so no need to implement its function
def gamma():
    # From Fig. 2 of the paper
    return 0.2


def k(time):
    return 0+time


# t = timestep --> in this case there is a single one
t = 0.1
# theta = np.arccos(np.exp(k(t)))
theta = np.pi
theta1 = np.pi
phi1 = np.pi
lambda1 = np.pi
theta2 = np.pi
phi2 = np.pi
lambda2 = np.pi

Q_env = QuantumRegister(1, 'Q_env')
Q_sys = QuantumRegister(1, 'Q_sys')
C = ClassicalRegister(1, 'C')

# Create a quantum circuit with two qubits
circuit = QuantumCircuit(Q_env, Q_sys, C)

# Initialize the second qubit in the desired state (1 = index of second qbit)
# circuit.initialize(desired_state, 1)

# Apply gates to circuit
# u = U3 gate
# ry = control-rotation-Y gate
# cnot = controlled-NOT gate
circuit.u(theta1, phi1, lambda1, 1)
circuit.ry(theta, 0)
circuit.cnot(0, 1)
circuit.u(theta2, phi2, lambda2, 1)

# Measurement
# first parameter = 1 = qbit on which the measurement takes place
# second parameter = 2 = classical bit to place the measurement result in
circuit.measure(Q_sys, C)

print(circuit.draw())
print(circuit.draw())
circuit.draw('mpl', filename='1t_circuit.png')
