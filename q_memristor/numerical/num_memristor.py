from scipy.integrate import quad
from sympy import symbols, diff
from q_memristor.numerical.operators import *
import numpy as np


"""
    Numerical Implementation of Quantum Memmristive Dynamics based on the article "Quantum Memristors with Quantum 
    Computers" by Y.-M. Guo, F. Albarr ́an-Arriagada, H. Alaeian, E. Solano, and G. Alvarado Barrios. 

    Author: Chiara Paglioni
"""


class memristor:

    def __init__(self, y0, w, h, m, a, b):
        self.y0 = y0
        self.w = w
        self.h = h
        self.m = m
        self.a = a
        self.b = b

    def gamma(self, ts):
        """
            Implementation of the decay rate where:
            - y0 = constant associated to the decay strength
            (Eq. 7 from "Quantum Memristors with Quantum Computers")
        """
        return self.y0 * (1 - np.sin(np.cos(self.w * ts)))

    def k1(self, ts):
        """
            Update of function k(t) based on timestep ts
            (Eq. 14 from "Quantum Memristors with Quantum Computers")
        """
        integ = lambda t_prime: self.gamma(t_prime)
        result = quad(integ, 0, ts)[0]
        return -result / 2

    def k(self, ts, ts_next):
        """
            Update of function k(t) based on timesteps t and t+1 where:
            - ts = lower bound of integral
            - ts_next = upper bound of integral
        """
        integ = lambda t_prime: self.gamma(t_prime)
        result = quad(integ, ts, ts_next)[0]
        return -result / 2

    def oscillatory_k(self, t1, t2):
        k_val = self.y0 * np.sin(self.w * (t1 + t2))
        if k_val > 0:
            return -k_val
        else:
            return k_val

    def get_density_mat(self, k):
        """
            Computes the density matrix p_I of a system at time t
            (Eq. 15 from "Quantum Memristors with Quantum Computers")
        """
        mat = np.empty((2, 2), dtype=complex)
        mat[0, 0] = (np.cos(self.a)*np.exp(k))**2
        mat[0, 1] = np.cos(self.a)*np.sin(self.a)*np.exp(-1j*self.b)*np.exp(k)
        mat[1, 0] = np.cos(self.a)*np.sin(self.a)*np.exp(1j*self.b)*np.exp(k)
        mat[1, 1] = 1-((np.cos(self.a)*np.exp(k))**2)
        return mat

    def get_density_mat2(self, ts):
        """
            Computes the density matrix p_I of a system at time t using the Krauss operators E_0 and E_1
            (Eq. 16 from "Quantum Memristors with Quantum Computers")
        """
        k0 = -self.y0/2
        e0 = self.get_E0_t(ts)
        e1 = self.get_E1_t(ts)
        return (e0 @ self.get_density_mat(k0) @ self.adjoint(e0)) + (e1 @ self.get_density_mat(k0) @ self.adjoint(e1))

    def get_E0_t(self, ts):
        mat = np.empty((2, 2), dtype=complex)
        mat[0, 0] = np.exp(self.k1(ts))
        mat[0, 1] = 0
        mat[1, 0] = 0
        mat[1, 1] = 1
        return mat

    def get_E1_t(self, ts):
        mat = np.empty((2, 2), dtype=complex)
        mat[0, 0] = 0
        mat[0, 1] = 0
        mat[1, 0] = np.sqrt((1-np.exp(2*self.k1(ts))))
        mat[1, 1] = 0
        return mat

    def get_E0(self, ts, ts_next):
        mat = np.empty((2, 2), dtype=complex)
        mat[0, 0] = np.exp(self.k(ts, ts_next))
        mat[0, 1] = 0
        mat[1, 0] = 0
        mat[1, 1] = 1
        return mat

    def get_E1(self, ts, ts_next):
        mat = np.empty((2, 2), dtype=complex)
        mat[0, 0] = 0
        mat[0, 1] = 0
        mat[1, 0] = np.sqrt((1-np.exp(2*self.k(ts, ts_next))))
        mat[1, 1] = 0
        return mat

    def exp_value(self, matrix, state_vector):
        """
            Returns the expectation value of a matrix and a state vector by taking the real part of their product
            <phi|mat|phi> = expectation value where:
            - phi = state vector
            - mat = matrix
        """
        expectation = np.vdot(state_vector, matrix @ state_vector)
        return expectation.real

    def trace_exp_value(self, matrix, state_vector):
        """
            Returns the expectation value of a matrix and a state vector by taking the trace of their product
        """
        exp = np.dot(state_vector, matrix)
        return np.trace(exp)

    def derivative(self, func, x):
        """
            Returns the derivative of a function (func) with respect to the variable x
            (Note that x should be a string)
        """
        t = symbols(x)
        f_derivative = diff(func, t)
        return f_derivative

    def anticomm(self, matrix1, matrix2):
        """
            Implementation of the anticommutator of two quantum operators
            {mat1, mat2} = mat1*mat2 + mat2*mat1
        """
        return np.dot(matrix1, matrix2) + np.dot(matrix2, matrix1)

    def comm(self, matrix1, matrix2):
        """
            Implementation of the commutator of two quantum operators
            [mat1, mat2] = mat1*mat2 - mat2*mat1
        """
        return np.dot(matrix1, matrix2) - np.dot(matrix2, matrix1)

    def hamiltonian(self):
        """
            Implementation of Hamiltonian of the system
            (Eq. 5 from "Quantum Memristors with Quantum Computers")
        """
        return (1/2)*self.h*self.w*(pauli_z2+2)

    def master_eq_I(self, ts, p_I):
        """
            Implementation of master equation of the system
            (Eq. 10 from "Quantum Memristors with Quantum Computers")
        """
        return self.gamma(ts)*((np.dot(np.dot(pauli_low2, p_I), pauli_ris2))-(self.anticomm(np.dot(pauli_ris2, pauli_low2), p_I)))

    def master_eq_2(self, ts, p_2):
        """
            Implementation of master equation of the system
            (Eq. 6 from "Quantum Memristors with Quantum Computers")
        """
        H = self.hamiltonian()
        return ((-1j/self.h)*self.comm(H, p_2))+self.gamma(ts)*((np.dot(np.dot(pauli_low2, p_2), pauli_ris2))-(self.anticomm(np.dot(pauli_ris2, pauli_low2), p_2)))

    def get_Schrodinger(self, t, p_I):
        """
            Translation of the density matrix p_I at time t into the Schrodinger picture
            i.e. it returns the corresponding p_2 at time t
        """
        return np.exp((-1j*t)/(self.h*self.hamiltonian())) * p_I * np.exp((1j*t)/(self.h*self.hamiltonian()))

    def adjoint(self, mat):
        """
            Returns the adjoint of a matrix, i.e. the transpose of its complex conjugate.
        """
        return np.conjugate(mat.T)
