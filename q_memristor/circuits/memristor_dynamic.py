from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RYGate
import numpy as np
from Simulator import IBMQSimulator

"""
    Simulation of dynamic quantum memristor 
    (Circuit shown in Fig. 3. of "Quantum Memristors with Quantum Computers")
"""


if __name__ == '__main__':

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

    pure_state = [np.cos(a), np.sin(a) * np.exp(1j * b)]
    zero_state = [1, 0]

    backend_string = 'qasm_simulator'
    shots = 50000

    # simulator = IBMQSimulator(backend_string, shots)
    simulator = IBMQSimulator(backend_string, shots)

    # REGISTERS OF THE CIRCUIT
    # Q_sys = memristor
    Q_env = QuantumRegister(len(t), 'Q_env')
    Q_sys = QuantumRegister(1, 'Q_sys')
    C = ClassicalRegister(1, 'C')

    # Create the quantum circuit
    circuit = QuantumCircuit(Q_env, Q_sys, C)

    # INITIALIZATION PROCESS
    # circuit.u(np.pi, np.pi, np.pi, Q_sys)
    # circuit.initialize(pure_state, Q_sys)
    # initial_state = [1] + [0] * (2 ** len(Q_env) - 1)
    # circuit.initialize(initial_state, Q_env)

    # EVOLUTION PROCESS
    for i in range(len(t) - 1):
        print('Time-step: ', t[i])

        theta = np.arccos(np.exp(simulator.k(t[i + 1], t[i])))
        print('Theta: ', theta)

        evol_qc = QuantumCircuit(Q_env, Q_sys, name='evolution')
        # Implementation of controlled-RY (cry) gate
        cry = RYGate(theta).control(1)
        # Apply cry gate to each timestep of the evolution
        for x in range(len(t)):
            evol_qc.append(cry, [Q_sys, Q_env[len(t) - 1 - x]])
            evol_qc.cnot(Q_env[len(t) - 1 - x], Q_sys)

        all_qbits = Q_env[:] + Q_sys[:]
        circuit.append(evol_qc, all_qbits)

    # MEASUREMENT PROCESS

    # TODO: change circuit to have two measurement process at the same time
    # These measurements provide the values of the matrices that should be plugged into the equation of
    # the voltage and current.
    circuit.sdg(Q_sys)      # Apply S gate to qubit 1
    circuit.h(Q_sys)        # Apply H gate to qubit 0

    # first parameter = 1 = qbit on which the measurement takes place
    # second parameter = 2 = classical bit to place the measurement result in
    circuit.measure(Q_sys, C)

    # UNCOMMENT TO DISPLAY CIRCUIT
    # print(circuit.draw())
    # print(circuit.decompose().draw())

    # Save image of final circuit
    # circuit.decompose().draw('mpl', filename='dynamic_circuit.png')

    # Execute the circuit using the simulator
    counts, measurements = simulator.execute_circuit(circuit)

    print('Simulator Measurement: ', counts)
    # Example Simulator Measurement:  {'1': 16, '0': 1008}
    # 1 --> obtained 16 times
    # 0 --> obtained 1008 times
    print('Measurements', measurements)

    # Define the observable matrix
    # observable_matrix = np.array([[0, -1j], [1j, 0]])   # Pauli-Y

    # Calculate the expectation value for each shot
    # expectation_value = 0
    # for outcome in counts:
    #     probability = counts[outcome] / shots
    #     measurement_result = int(outcome, 2)  # Convert outcome (binary string) to integer
    #     expectation_value += probability * observable_matrix[measurement_result][measurement_result]
    #
    # print("Exp value: ")
    # print(expectation_value)

    # # EXPECTATION VALUES
    # # num_shots = sum(counts.values())
    # # expectation_values = {}
    # #
    # # for register in counts:
    # #     expectation_values[register] = counts[register] / num_shots
    # #
    # # print("Expectation values:", expectation_values, '\n')
