from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit.library import RYGate
import numpy as np
from scipy.integrate import quad

from q_memristor.circuits.t_plot_circuit import Tplot

"""
    Circuit simulation of dynamic quantum memristor based on the article "Quantum Memristors with Quantum 
    Computers" from Y.-M. Guo, F. Albarr ́an-Arriagada, H. Alaeian, E. Solano, and G. Alvarado Barrios. 

    Author: Chiara Paglioni
    Link to Article: https://link.aps.org/doi/10.1103/PhysRevApplied.18.024082  
"""


class IBMQSim:
    def __init__(self, backend, shots, y0, w):
        self.backend = backend
        self.shots = shots
        self.y0 = y0
        self.w = w

    def execute_circuit(self, circ):
        # shots = 50000 (according to paper) --> reduced for computation purposes
        sim = Aer.get_backend(self.backend)
        job = execute(circ, sim, shots=self.shots)
        result = job.result()
        cnts = result.get_counts(circ)
        exp_value = (cnts['0'] - cnts['1']) / self.shots
        return cnts, exp_value,

    def gamma(self, ts):
        return self.y0 * (1 - np.sin(np.cos(self.w * ts)))

    def gamma_int(self, ts):
        # return self.y0 * ts - self.y0 * ((np.sin(np.cos(ts))) / (np.sin(ts))) * np.cos(ts)
        return self.y0 * (np.cos(ts) + ts)

    def k(self, t, t_next):
        # gamma_t = self.gamma_int(t)
        # gamma_next = self.gamma_int(t_next)
        # return -(gamma_next - gamma_t)/2
        return -self.gamma(t_next)

    def gamma_og(self, ts):
        return self.y0 * (1 - np.sin(np.cos(self.w * ts)))

    def k_og(self, ts_next, ts):
        integrand = lambda t_prime: self.gamma_og(t_prime)
        print('Function: ', str(integrand))
        integral_result, _ = quad(integrand, ts, ts_next)
        return -integral_result/2


if __name__ == '__main__':

    # Time-steps
    # Number of time steps = (1 - 0) / 0.0333 seconds = 30 time steps
    # eps = 0.0333
    eps = 0.1
    tmax = 1.1
    t = np.arange(0, tmax, eps)

    # SIMULATION PARAMETERS
    # a and b are the parameters used in the pure state used to initialize the memristor
    # Based on Fig. 2 of the paper:
    # y0 = 0.2 or 0.02
    a0 = np.pi / 4
    b0 = np.pi / 5
    y0 = 0.4
    k0 = -y0/2
    w = 1
    m = 1
    h = 1

    pure_state = [np.cos(a0), np.sin(a0) * np.exp(1j * b0)]

    backend_string = 'qasm_simulator'
    shots = 500

    simulator = IBMQSim(backend_string, shots, y0, w)

    # iv_plot = IVplot()
    t_plot = Tplot()

    V = []
    I = []

    # REGISTERS OF THE CIRCUIT
    # Q_sys = memristor
    Q_env = QuantumRegister(len(t), 'Q_env')
    Q_sys = QuantumRegister(1, 'Q_sys')
    C = ClassicalRegister(1, 'C')

    # Create the quantum circuit
    circuit = QuantumCircuit(Q_env, Q_sys, C)

    # INITIALIZATION PROCESS
    circuit.initialize(pure_state, Q_sys)

    # The other registers are automatically initialized to the zero state
    zero_state = [1] + [0] * (2 ** len(Q_env) - 1)
    circuit.initialize(zero_state, Q_env)

    # EVOLUTION PROCESS
    # expectation_values = []
    expectation_values = [0.6, 0.8, 0.5, 0.01, -0.1, -0.2, -0.5, -0.6, -0.8, -0.86]
    sim_counts = []
    thetas = []
    amplitudes = []

    theta0 = np.arccos(np.exp(k0))
    thetas.append(theta0)

    x = 0
    for i in range(0, len(t) - 1):
        x += 1

        print('Time-step: ', t[i])
        print('Theta: ', thetas[i])

        # Implementation of controlled-RY (cry) gate
        cry = RYGate(thetas[i]).control(1)

        evol_qc = QuantumCircuit(Q_env, Q_sys, name='evolution')
        # Apply cry gate to each timestep of the evolution
        evol_qc.append(cry, [Q_sys, Q_env[i - x]])
        evol_qc.cnot(Q_env[i - x], Q_sys)

        all_qbits = Q_env[:] + Q_sys[:]
        circuit.append(evol_qc, all_qbits)

        # MEASUREMENT PROCESS

        # The measurement over Pauli-Y provides the values of the matrices that should be plugged into the equation of
        # the voltage and current.
        circuit.sdg(Q_sys)
        circuit.h(Q_sys)

        # first parameter = 1 = qbit on which the measurement takes place
        # second parameter = 2 = classical bit to place the measurement result in
        circuit.measure(Q_sys, C)

        # UNCOMMENT TO DISPLAY CIRCUIT
        # print(circuit.draw())
        # print(circuit.decompose().draw())

        # Save image of final circuit
        # circuit.decompose().draw('mpl', filename='dynamic_circuit.png')

        # Execute the circuit using the simulator
        counts, exp_value = simulator.execute_circuit(circuit)

        theta = np.arccos(np.exp(simulator.k(t[i], t[i+1])))
        # theta = np.exp(simulator.k(t[i], t[i+1]))
        print('K at t ', t[i], ' and ', t[i+1], ' : ', simulator.k_og(t[i+1], t[i]))
        print('K at t ', t[i], ' and ', t[i + 1], ' : ', simulator.k(t[i], t[i+1]))

        thetas.append(theta)
        sim_counts.append(counts)

        print('Simulator Measurement: ', counts)

        # expectation_values.append(exp_value)
        # print('Expectation Value: ', exp_value)

        V.append(-(1 / 2) * np.sqrt((m * h * w) / 2) * expectation_values[i])
        I.append(simulator.gamma(t[i]) * V[i])
        print('Gamma at time ', t[i], ' : ', simulator.gamma(t[i]))
        print('Voltage at time ', t[i], ' : ', V[i])
        print('Current at time ', t[i], ' : ', I[i])
        print()

        t_plot.update(t[i], V[i], I[i])

    t_plot.save_plot()
