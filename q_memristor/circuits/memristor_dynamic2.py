from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RYGate
import numpy as np
from simulator import IBMQSimulator
from q_memristor.numerical.num_memristor import memristor
from q_memristor.circuits.t_plot_circuit import Tplot

"""
    Circuit simulation of dynamic quantum memristor based on the article "Quantum Memristors with Quantum 
    Computers" from Y.-M. Guo, F. Albarr ́an-Arriagada, H. Alaeian, E. Solano, and G. Alvarado Barrios. 

    Author: Chiara Paglioni
    Link to Article: https://link.aps.org/doi/10.1103/PhysRevApplied.18.024082  
    
    CIRCUIT SIMULATION: 
    A a new circuit is created at each time-step increase and the number of Q_env register is also increased at each 
    timestep. 
"""

# Time-steps
# Number of time steps = (1 - 0) / 0.0333 seconds = 30 time steps
eps = 0.1
tmax = 1.1
t = np.arange(0, tmax, eps)

# SIMULATION PARAMETERS
# Based on Fig. 2 of the paper --> y0 = 0.2 or 0.02
a = np.pi / 4
b = np.pi / 5
y0 = 0.4
w = 1
m = 1
h = 1

pure_state = [np.cos(a), np.sin(a) * np.exp(1j * b)]

backend_string = 'qasm_simulator'
shots = 50000

sim = IBMQSimulator(backend_string, shots)

# iv_plot = IVplot()
t_plot = Tplot()

V = []
I = []

if __name__ == '__main__':

    mem = memristor(y0, w, h, m, a, b)

    # EVOLUTION PROCESS
    expectation_values = []
    sim_counts = []
    for i in range(len(t) - 1):
        # REGISTERS OF THE CIRCUIT
        # Q_sys = memristor
        Q_env = QuantumRegister(i+1, 'Q_env')
        Q_sys = QuantumRegister(1, 'Q_sys')
        C = ClassicalRegister(1, 'C')

        circuit = QuantumCircuit(Q_env, Q_sys, C)

        # INITIALIZATION PROCESS
        circuit.initialize(pure_state, Q_sys)

        # The other registers are automatically initialized to the zero state
        zero_state = [1] + [0] * (2 ** len(Q_env) - 1)
        circuit.initialize(zero_state, Q_env)

        print('Time-step: ', t[i])

        if i == 0:
            theta = np.arccos(np.exp(mem.k(t[0], t[1])))
            print('Theta: ', theta, ' at time: ', 0.0)

            # Implementation of controlled-RY (cry) gate
            cry = RYGate(theta).control(1)

            circuit.append(cry, [Q_sys, Q_env])
            circuit.cnot(Q_env, Q_sys)

        else:
            evol_qc = QuantumCircuit(Q_env, Q_sys, name='evolution')
            # Apply cry gate to each timestep of the evolution
            for j in range(i+1):
                # theta = np.arccos(np.exp(mem.k(t[j], t[j + 1])))
                # print('K: ', mem.k(t[j], t[j + 1]))
                print('Theta: ', theta, ' at time: ', t[j])

                # Implementation of controlled-RY (cry) gate
                cry = RYGate(theta).control(1)

                evol_qc.append(cry, [Q_sys, Q_env[i - j]])
                evol_qc.cnot(Q_env[i - j], Q_sys)

            all_qbits = Q_env[:] + Q_sys[:]
            circuit.append(evol_qc.to_instruction(), all_qbits)

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
        # circuit.decompose().draw('mpl', filename='figures/dynamic_circuit2.png')

        # Execute the circuit using the simulator
        counts, measurements, exp_value = sim.execute_circuit(circuit)
        sim_counts.append(counts)

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

    t_plot.save_plot('figures/plot_dynamic_circuit2.png')

    # TEST EXPECTATION VALUE
    # Calculate expectation values increasing the number of shots at each iteration
    sim_expectation = []
    shot_val = 0
    zero_count = 0
    one_count = 0
    for count in sim_counts:
        shot_val += shots
        zero_count += count['0']
        one_count += count['1']
        expect_value = (zero_count - one_count) / shot_val
        sim_expectation.append(expect_value)

    print('Expectation Values: ')
    print(sim_expectation)
