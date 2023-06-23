from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RYGate
import numpy as np

from q_memristor.circuits.csv_generator import csv_gen
from q_memristor.circuits.simulator import IBMQSimulator
from q_memristor.circuits.t_plot_circuit import Tplot
from q_memristor.numerical.num_memristor import memristor

"""
    Circuit simulation of two coupled memristors based on the article "Quantum Memristors with Quantum 
    Computers" from Y.-M. Guo, F. Albarr ́an-Arriagada, H. Alaeian, E. Solano, and G. Alvarado Barrios. 

    Author: Chiara Paglioni
    Link to Article: https://link.aps.org/doi/10.1103/PhysRevApplied.18.024082  
"""

# Time-steps
eps = 0.1
tmax = 1.1
t = np.arange(0, tmax, eps)

# SIMULATION PARAMETERS
# Note that in theis simulation the values of a and b are the amplitudes of the initial state directly
a = 1/np.sqrt(2)
b = 1/np.sqrt(2)
y0 = 0.02
w = 1
m = 1
h = 1
delta = 1

pure_state = np.array([a, b], dtype=complex)

backend_string = 'qasm_simulator'
shots = 50000

sim = IBMQSimulator(backend_string, shots)

# iv_plot = IVplot()
t_plot = Tplot()

V = []
I = []
thetas = np.empty((len(t)))

if __name__ == '__main__':

    mem = memristor(y0, w, h, m, a, b)

    # EVOLUTION PROCESS
    expectation_values = []
    sim_counts = []
    for i in range(len(t) - 1):

        # REGISTERS OF THE CIRCUIT
        # Q_sys = memristor
        Q_env1 = QuantumRegister(i+1, 'Q_env1')
        Q_sys1 = QuantumRegister(1, 'Q_sys1')
        Q_env2 = QuantumRegister(i+1, 'Q_env2')
        Q_sys2 = QuantumRegister(1, 'Q_sys2')
        C = ClassicalRegister(1, 'C')

        # Create the quantum circuit
        circuit = QuantumCircuit(Q_env1, Q_sys1, Q_sys2, Q_env2, C)

        # INITIALIZATION PROCESS
        circuit.initialize(pure_state, Q_sys1)
        circuit.initialize(pure_state, Q_sys2)
        zero_state = [1] + [0] * (2 ** len(Q_env1) - 1)
        circuit.initialize(zero_state, Q_env1)
        circuit.initialize(zero_state, Q_env2)

        print('Time-step: ', t[i])

        # EVOLUTION PROCESS
        for j in range(i+1):
            evol_qc = QuantumCircuit(Q_env1, Q_sys1, Q_sys2, Q_env2, name='evolution')

            theta = np.arccos(np.exp(mem.k(t[j], t[j + 1])))
            print('Theta: ', theta, ' at time: ', t[j])
            thetas[j] = theta

            # Implementation of controlled-RY (cry) gate
            cry1 = RYGate(theta).control(1)
            cry2 = RYGate(theta).control(1)

            # Apply evolution process and interaction operator for each time-step
            evol_qc.append(cry1, [Q_sys1, Q_env1[i - j]])
            evol_qc.append(cry2, [Q_sys2, Q_env2[j]])
            evol_qc.cnot(Q_env1[i - j], Q_sys1)
            evol_qc.cnot(Q_sys2, Q_env2[j])

            # The following are the operators that form the interaction operator
            interaction_qc = QuantumCircuit(Q_sys1, Q_sys2, name='interaction')
            interaction_qc.rx(np.pi/2, Q_sys1)
            interaction_qc.rx(np.pi/2, Q_sys2)
            interaction_qc.cnot(Q_sys1, Q_sys2)
            interaction_qc.rz(2*delta, Q_sys2)
            interaction_qc.cnot(Q_sys1, Q_sys2)
            interaction_qc.rx(-np.pi/2, Q_sys1)
            interaction_qc.rx(-np.pi/2, Q_sys2)
            evol_qc.append(interaction_qc.to_instruction(), [Q_sys1, Q_sys2])

            all_qbits = Q_env1[:] + Q_sys1[:] + Q_sys2[:] + Q_env2[:]
            circuit.append(evol_qc.to_instruction(), all_qbits)

        # MEASUREMENT PROCESS
        # Transform into Pauli-Y measurement
        circuit.sdg(Q_sys1)
        circuit.sdg(Q_sys2)
        circuit.h(Q_sys1)
        circuit.h(Q_sys2)
        circuit.measure(Q_sys1, C)
        circuit.measure(Q_sys2, C)

        # UNCOMMENT TO DISPLAY CIRCUIT
        # print(circuit.draw())
        # print(circuit.decompose().draw())

        # Save image of final circuit
        circuit.decompose().draw('mpl', filename='figures/coupled_circuit.png')

        # Execute the circuit using the simulator
        counts, measurements, exp_value = sim.execute_circuit(circuit)
        sim_counts.append(counts)
        expectation_values.append(exp_value)

        print('Simulator Measurement: ', counts)
        print("Expectation values:", exp_value)

        V.append(-(1 / 2) * np.sqrt((m * h * w) / 2) * expectation_values[i])
        I.append(mem.gamma(t[i]) * V[i])
        print('Gamma at time ', t[i], ' : ', mem.gamma(t[i]))
        print('Voltage at time ', t[i], ' : ', V[i])
        print('Current at time ', t[i], ' : ', I[i])
        print()

        # iv_plot.update(V[i], I[i])
        t_plot.update(t[i], V[i], I[i])

    t_plot.save_plot('figures/plot_coupled_circuit.png')

    # Save data into csv file
    csvGen = csv_gen('data/data_mem_coupled.csv')
    csvGen.write_data(t, V, I, expectation_values, thetas)
