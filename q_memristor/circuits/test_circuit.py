from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
from qiskit.circuit.library import RYGate
from q_memristor.circuits.Simulator import IBMQSimulator
from q_memristor.circuits.iv_plot_circuit import IVplot
from q_memristor.numerical.num_memristor import memristor
from q_memristor.plots.time_plot import Tplot

"""
    TEST 1
    A new circuit is created where a new evolution step is added for each time step
    
    Results: 
        - While the initial results seem to be in line with the one showed in the paper, 
        the update does not behave according to the same behaviour shown in the paper 
        (this might be due to the wrong value of the time?)
"""

if __name__ == '__main__':

    a = np.pi / 4
    b = np.pi / 5
    y0 = 0.4
    w = 1
    m = 1
    h = 1
    backend_string = 'qasm_simulator'
    shots = 50000

    pure_state = [np.cos(a), np.sin(a) * np.exp(1j * b)]

    mem = memristor(y0, w, h, m, a, b)
    simulator = IBMQSimulator(backend_string, shots)

    iv_plot = IVplot()
    t_plot = Tplot()

    eps = 0.1
    tmax = 1.1
    t = np.arange(0, tmax, eps)

    V = []
    I = []

    n = 0
    expectation_values = []
    for i in range(len(t)):
        print('Time-step: ', t[i])

        n += 1

        # Initialize registers
        Q_env = QuantumRegister(n, 'Q_env')
        Q_sys = QuantumRegister(1, 'Q_sys')
        C = ClassicalRegister(1, 'C')

        # Create a quantum circuit with two qubits
        circuit = QuantumCircuit(Q_env, Q_sys, C)

        # Apply gates to circuit

        # INITIALIZATION
        zero_state = [1] + [0] * (2 ** len(Q_env) - 1)

        circuit.initialize(zero_state, Q_env)
        circuit.initialize(pure_state, Q_sys)

        # EVOLUTION
        x = 0
        for j in range(n):
            if n == 1:
                theta = np.arccos(np.exp(mem.k1(t[j])))
            else:
                theta = np.arccos(np.exp(simulator.k(t[j + 1], t[j])))
            print('Theta ', x, ' : ', theta)

            # Implementation of controlled RY gate
            cry = RYGate(theta).control(1)

            evol_qc = QuantumCircuit(Q_env, Q_sys, name='evolution')
            # Apply cry gate to each timestep of the evolution
            if n == 1:
                circuit.append(cry, [Q_sys, Q_env])
                circuit.cnot(Q_env, Q_sys)
            else:
                x += 1
                evol_qc.append(cry, [Q_sys, Q_env[n - 1 - x]])
                evol_qc.cnot(Q_env[n - 1 - x], Q_sys)

            all_qbits = Q_env[:] + Q_sys[:]
            circuit.append(evol_qc, all_qbits)

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

        # Results for initial values at t = 0
        #
        # Simulator Expectation Value:  0.5888
        # Voltage:  -0.2081722363813196
        # Current:  -0.013426173789837254
        v_val = -(1 / 2) * np.sqrt((m * h * w) / 2) * exp_value
        i_val = mem.gamma(t[i]) * v_val

        iv_plot.update(v_val, i_val)
        t_plot.update(t[i], v_val, i_val)

        V.append(v_val)
        I.append(i_val)

        print('Voltage: ', v_val)
        print('Current: ', i_val)
        print()
