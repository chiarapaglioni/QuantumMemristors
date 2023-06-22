import numpy as np
from qiskit import Aer, execute


class IBMQSimulator:
    def __init__(self, backend, shots):
        self.backend = backend
        self.shots = shots

    def execute_circuit(self, circ):
        # shots = 50000 (according to paper) --> reduced for computation purposes
        sim = Aer.get_backend(self.backend)
        job = execute(circ, sim, shots=self.shots, memory=True)
        result = job.result()
        measurements = result.get_memory()
        cnts = result.get_counts(circ)
        exp_value = np.abs(cnts['0'] - cnts['1']) / self.shots
        return cnts, measurements, exp_value
