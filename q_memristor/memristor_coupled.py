from qiskit import Aer, execute, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RYGate
from scipy.integrate import quad
import numpy as np

"""
    Simulation of circuit simulating two coupled memristors 
    (Circuit shown in Fig. 5. and Fig. 6. of "Quantum Memristors with Quantum Computers")
"""


class IBMQSimulator:
    def __init__(self, backend='qasm_simulator'):
        self.backend = backend

    def execute_circuit(self, circ, shots=1000):
        # shots = 50000 (according to paper) --> reduced for computation purposes
        sim = Aer.get_backend(self.backend)
        job = execute(circ, sim, shots=shots)
        result = job.result()
        cnts = result.get_counts(circ)
        return cnts


# TODO: check correct implementation of decay rate function
def gamma(y0, w, ts):
    return y0 * (1 - np.sin(np.cos(w * ts)))


# TODO: check correct implementation of k function
def k(ts_next, ts):
    integrand = lambda t_prime: gamma(y0, w, t_prime)
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
Q_env1 = QuantumRegister(len(t), 'Q_env1')
Q_sys1 = QuantumRegister(1, 'Q_sys1')
Q_env2 = QuantumRegister(len(t), 'Q_env2')
Q_sys2 = QuantumRegister(1, 'Q_sys2')
C = ClassicalRegister(1, 'C')

# Create the quantum circuit
circuit = QuantumCircuit(Q_env1, Q_sys1, Q_sys2, Q_env2, C)

# INITIALIZATION PROCESS
# circuit.u(theta1, phi1, lambda1, Q_sys)
circuit.initialize(pure_state, Q_sys1)
circuit.initialize(pure_state, Q_sys2)
initial_state = [1] + [0] * (2 ** len(Q_env1) - 1)
circuit.initialize(initial_state, Q_env1)
initial_state = [1] + [0] * (2 ** len(Q_env2) - 1)
circuit.initialize(initial_state, Q_env2)

# EVOLUTION PROCESS
for i in range(len(t) - 1):
    print('Time-step: ', t[i])

    theta = np.arccos(np.exp(k(t[i + 1], t[i])))
    print('Theta: ', theta)

    evol_qc = QuantumCircuit(Q_env1, Q_sys1, Q_sys2, Q_env2, name='evolution')
    # Implementation of controlled-RY (cry) gate
    cry1 = RYGate(theta).control(1)
    cry2 = RYGate(theta).control(1)
    # Apply cry gate to each timestep of the evolution
    for x in range(len(t)):
        evol_qc.append(cry1, [Q_sys1, Q_env1[len(t) - 1 - x]])
        evol_qc.append(cry2, [Q_sys2, Q_env2[x]])
        evol_qc.cnot(Q_env1[len(t) - 1 - x], Q_sys1)
        evol_qc.cnot(Q_sys2, Q_env2[x])

    all_qbits = Q_env1[:] + Q_sys1[:] + Q_sys2[:] + Q_env2[:]
    circuit.append(evol_qc.to_instruction(), all_qbits)

    # MEASUREMENT PROCESS
    # first parameter = 1 = qbit on which the measurement takes place
    # second parameter = 2 = classical bit to place the measurement result in
    circuit.u(theta2, phi2, lambda2, Q_sys1)
    circuit.u(theta2, phi2, lambda2, Q_sys2)
    circuit.measure(Q_sys1, C)
    circuit.measure(Q_sys2, C)

    # UNCOMMENT TO DISPLAY CIRCUIT
    # print(circuit.draw())
    # print(circuit.decompose().draw())

    # Save image of final circuit
    circuit.decompose().draw('mpl', filename='coupled_circuit.png')

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