from qiskit import Aer, execute, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RYGate
from scipy.integrate import quad
import numpy as np


"""
    Simulation of two coupled quantum memristors
    (Circuit shown in Fig. 5. and 6. of "Quantum Memristors with Quantum Computers")
"""


class IBMQSimulator:
    def __init__(self, backend='qasm_simulator'):
        self.backend = backend

    def execute_circuit(self, circ, shots=1024):
        sim = Aer.get_backend(self.backend)
        job = execute(circ, sim, shots=shots)
        result = job.result()
        cnts = result.get_counts(circ)
        return cnts
