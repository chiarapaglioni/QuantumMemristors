import numpy as np
from scipy.integrate import quad
from sympy import symbols, diff


class memristor:

    def __init__(self, y0, w, h, m, a, b):
        self.y0 = y0
        self.w = w
        self.h = h
        self.m = m
        self.a = a
        self.b = b

    # Note that --> gamma has a sinusoidal time dependance
    # Sinusoidal time dependence = quantity or phenomenon that varies with time in a sinusoidal/harmonic manner
    # Characterized by a periodic oscillation that follows a sine or cosine function.
    def gamma(self, ts):
        return self.y0 * (1 - np.sin(np.cos(self.w * ts)))

    def k(self, ts_next, ts):
        result, _ = quad(self.gamma, ts, ts_next)
        return -result / 2

    def p(self, k):
        mat = np.empty((2, 2), dtype='complex_')
        mat[0, 0] = (np.cos(self.a)*np.exp(k))**2
        mat[0, 1] = np.cos(self.a)*np.sin(self.a)*np.exp(-1j*self.b)*np.exp(k)
        mat[1, 0] = np.cos(self.a)*np.sin(self.a)*np.exp(1j*self.b)*np.exp(k)
        mat[1, 1] = 1-((np.cos(self.a)*np.exp(k))**2)
        return mat

    def exp_value(self, pauli_matrix, state_vector):
        """Compute the expectation value of a Pauli matrix.

        Args:
            pauli_matrix (str): The Pauli matrix to compute the expectation value for.
                                Can be 'X', 'Y', or 'Z'.
            state_vector (numpy.ndarray): The quantum state vector.

        Returns:
            float: The expectation value.
        """
        pauli_operator = pauli_matrix
        expectation = np.vdot(state_vector, pauli_operator @ state_vector)
        return expectation.real

    def derivative(self, func):
        # Define the variable
        t = symbols('t')
        # Define the function
        # func = t ** 2 + 3 * t + 1
        # Take the derivative with respect to t
        f_derivative = diff(func, t)
        print('Derviative: ', f_derivative)
        return f_derivative

    # Anticommutator
    def anticomm(self, matrix1, matrix2):
        return np.dot(matrix1, matrix2) + np.dot(matrix2, matrix1)

    # Commutator
    def comm(self, matrix1, matrix2):
        return np.dot(matrix1, matrix2) - np.dot(matrix2, matrix1)

    def hamiltonian(self, p_z):
        p_temp = np.array(p_z)+2
        return (self.h*self.w*p_temp)/2
