from qiskit import Aer, execute, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RYGate
from scipy.integrate import quad
import numpy as np

"""
    Simulation of dynamic quantum memristor 
    (Circuit shown in Fig. 3. of "Quantum Memristors with Quantum Computers")
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


# TODO: check correct implementation of decay rate function
def gamma(ts):
    y0 = 0.2
    w = 1
    return y0 * (1 - np.sin(np.cos(w * ts)))


# TODO: check correct implementation of k function
def k(ts_next, ts):
    integrand = lambda t_prime: gamma(t_prime)
    integral_result, _ = quad(integrand, ts, ts_next)
    return -integral_result / 2


# Time-steps
eps = 0.1
tmax = 1
t = np.arange(0, tmax, eps)

# SIMULATION PARAMETERS
# TODO: determine correct parameters for the simulation
# a and b are the parameters used in the pure state used to initialize the memristor
#
# Based on Fig. 2 of the paper:
#   y0 = 0.2 or 0.02
#   w = 1
a = np.pi / 4
b = np.pi / 5
y0 = 0.2
w = 1
theta1 = np.pi
phi1 = np.pi
lambda1 = np.pi
theta2 = np.pi
phi2 = np.pi
lambda2 = np.pi

pure_state = [np.cos(a), np.sin(a) * np.exp(1j * b)]
zero_state = [1, 0]

# REGISTERS OF THE CIRCUIT
# Q_sys = memristor
Q_env = QuantumRegister(len(t), 'Q_env')
Q_sys = QuantumRegister(1, 'Q_sys')
C = ClassicalRegister(1, 'C')

# Create the quantum circuit
circuit = QuantumCircuit(Q_env, Q_sys, C)

# INITIALIZATION PROCESS
# circuit.u(theta1, phi1, lambda1, Q_sys)
circuit.initialize(pure_state, Q_sys)
initial_state = [1] + [0] * (2 ** len(Q_env) - 1)
circuit.initialize(initial_state, Q_env)

# EVOLUTION PROCESS
for i in range(len(t) - 1):
    print('Time-step: ', t[i])

    theta = np.arccos(np.exp(k(t[i + 1], t[i])))
    print('Theta: ', theta)

    evol_qc = QuantumCircuit(Q_env, Q_sys, name='evolution')
    # Implementation of controlled-RY (cry) gate
    cry = RYGate(theta).control(1)
    # Apply cry gate to each timestep of the evolution
    for x in range(len(t)):
        evol_qc.append(cry, [Q_sys, Q_env[len(t) - 1 - x]])
        evol_qc.cnot(Q_env[len(t) - 1 - x], Q_sys)

    all_qbits = Q_env[:] + Q_sys[:]
    circuit.append(evol_qc.to_instruction(), all_qbits)

# MEASUREMENT PROCESS
# first parameter = 1 = qbit on which the measurement takes place
# second parameter = 2 = classical bit to place the measurement result in
circuit.u(theta2, phi2, lambda2, Q_sys)
circuit.measure(Q_sys, C)

# UNCOMMENT TO DISPLAY CIRCUIT
# print(circuit.draw())
# print(circuit.decompose().draw())

# Save image of final circuit
circuit.decompose().draw('mpl', filename='dynamic_circuit.png')

# Execute the circuit using the simulator
simulator = IBMQSimulator()
counts = simulator.execute_circuit(circuit)

print('Simulator Measurement: ', counts)
# Example Simulator Measurement:  {'1': 16, '0': 1008}
# 1 --> obtained 16 times
# 0 --> obtained 1008 times

# EXPECTATION VALUES
num_shots = sum(counts.values())
expectation_values = {}

for register in counts:
    expectation_values[register] = counts[register] / num_shots

print("Expectation values:", expectation_values, '\n')
