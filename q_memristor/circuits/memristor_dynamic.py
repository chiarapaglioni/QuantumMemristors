from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RYGate
import numpy as np
from Simulator import IBMQSimulator
from q_memristor.circuits.iv_plot_circuit import IVplot
from q_memristor.numerical.num_memristor import memristor
from q_memristor.circuits.t_plot_circuit import Tplot

"""
    Circuit simulation of dynamic quantum memristor based on the article "Quantum Memristors with Quantum 
    Computers" from Y.-M. Guo, F. Albarr ́an-Arriagada, H. Alaeian, E. Solano, and G. Alvarado Barrios. 

    Author: Chiara Paglioni
    Link to Article: https://link.aps.org/doi/10.1103/PhysRevApplied.18.024082  
"""

if __name__ == '__main__':

    # Time-steps
    # Number of time steps = (1 - 0) / 0.0333 seconds = 30 time steps
    eps = 0.1
    tmax = 1.2
    t = np.arange(0, tmax, eps)

    # SIMULATION PARAMETERS
    # a and b are the parameters used in the pure state used to initialize the memristor
    # Based on Fig. 2 of the paper:
    # y0 = 0.2 or 0.02
    a = np.pi / 4
    b = np.pi / 5
    y0 = 0.4
    w = 1
    m = 1
    h = 1

    pure_state = [np.cos(a), np.sin(a) * np.exp(1j * b)]

    backend_string = 'qasm_simulator'
    shots = 50000

    # simulator = IBMQSimulator(backend_string, shots)
    simulator = IBMQSimulator(backend_string, shots)

    # iv_plot = IVplot()
    t_plot = Tplot()

    V = []
    I = []

    mem = memristor(y0, w, h, m, a, b)

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
    # zero_state = [1] + [0] * (2 ** len(Q_env) - 1)
    # print(zero_state)
    # circuit.initialize(zero_state, Q_env)

    # EVOLUTION PROCESS
    x = 0
    expectation_values = []
    for i in range(len(t) - 1):
        print('Time-step: ', t[i])

        # theta = np.arccos(np.exp(simulator.k(t[i + 1], t[i])))
        theta = np.arccos(np.exp(t[i]))
        print('Theta: ', theta)

        evol_qc = QuantumCircuit(Q_env, Q_sys, name='evolution')
        # Implementation of controlled-RY (cry) gate
        cry = RYGate(theta).control(1)
        # Apply cry gate to each timestep of the evolution
        x += 1
        evol_qc.append(cry, [Q_sys, Q_env[len(t) - 1 - x]])
        evol_qc.cnot(Q_env[len(t) - 1 - x], Q_sys)

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
        counts, measurements, exp_value = simulator.execute_circuit(circuit)

        print('Simulator Measurement: ', counts)
        # Example Simulator Measurement:  {'1': 16, '0': 1008}
        # 1 --> obtained 16 times
        # 0 --> obtained 1008 times
        # print('Measurements', measurements)
        expectation_values.append(exp_value)
        print('Expectation Value: ', exp_value)

        V.append(-(1 / 2) * np.sqrt((m * h * w) / 2) * expectation_values[i])
        I.append(mem.gamma(t[i]) * V[i])
        print('Gamma at time ', t[i], ' : ', mem.gamma(t[i]))
        print('Voltage at time ', t[i], ' : ', V[i])
        print('Current at time ', t[i], ' : ', I[i])
        print()

        # iv_plot.update(V[i], I[i])
        t_plot.update(t[i], V[i], I[i])

    t_plot.save_plot()

    # EXPECTATION VALUES
    # y_eigenvalues = {'0': 1, '1': -1}
    # pauli_y = np.array([[0, -1j], [1j, 0]])
    #
    # Calculate the expectation value for each shot
    # expectation_values = []
    # expectation_value = 0
    # count_1 = 0
    # count_0 = 0
    # shot_count = 0
    # for measurement in measurements:
    #     # Get the eigenvalue corresponding to the measurement outcome
    #     eigenvalue = y_eigenvalues[measurement]
    #
    #     shot_count = shot_count + 1
    #
    #     if int(measurement) == 1:
    #         count_1 = count_1 + 1
    #     else:
    #         count_0 = count_0 + 1
    #
    #     # Calculate the expectation value for the shot
    #     # expectation_value = expectation_value - (eigenvalue * int(measurement) * counts[measurement] / shots)
    #     expectation_value = np.abs(count_0 - count_1) / shot_count
    #     # print('Exp: ', expectation_value)
    #     expectation_values.append(expectation_value)

    # print('Expectation Values: ', expectation_values)

    # VOLTAGE CALCULATION
    # Initial results:
    # V0 = -0.3535533905932738
    # I0 = -0.011401302470989517
    # for i in range(len(expectation_values)):
        # iv_plot.update(V[i], I[i])
        # t_plot.update(ts, V[i], I[i])
        # print('Voltage at time ', ts, ' : ', V[i])
        # print('Current at time ', ts, ' : ', I[i])
