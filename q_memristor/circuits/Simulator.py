import numpy as np
from qiskit import Aer, execute
from scipy.integrate import quad


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
        return cnts, measurements

    def gamma(self, ts):
        y0 = 0.2
        w = 1
        return y0 * (1 - np.sin(np.cos(w * ts)))

    def k(self, ts_next, ts):
        integrand = lambda t_prime: self.gamma(t_prime)
        integral_result, _ = quad(integrand, ts, ts_next)
        return -integral_result / 2

