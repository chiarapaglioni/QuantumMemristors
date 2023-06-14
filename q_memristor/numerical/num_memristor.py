from q_memristor.plots import iv_plot, time_plot
from scipy.integrate import quad
from sympy import symbols, diff
from q_memristor.numerical.operators import *
import numpy as np


"""
    Numerical Implementation of Quantum Memristive Dynamics based on the article "Quantum Memristors with Quantum 
    Computers" from Y.-M. Guo, F. Albarr ́an-Arriagada, H. Alaeian, E. Solano, and G. Alvarado Barrios. 

    Author: Chiara Paglioni
    Link to Article: https://link.aps.org/doi/10.1103/PhysRevApplied.18.024082  
"""

# Questions:
# - When computing the exp_value at each iteration, do we multiply by a new pure state or always by the initial one?


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

    def k1(self, ts):
        result, _ = quad(self.gamma, 0, ts)
        return -result / 2

    def k(self, ts, ts_next):
        result, _ = quad(self.gamma, ts, ts_next)
        return -result / 2

    # Computes the density matrix of a system at time t
    # Implements Eq. 15 in "Quantum Memristors with Quantum Computers"
    def get_density_mat(self, k):
        mat = np.empty((2, 2), dtype='complex_')
        mat[0, 0] = (np.cos(self.a)*np.exp(k))**2
        mat[0, 1] = np.cos(self.a)*np.sin(self.a)*np.exp(-1j*self.b)*np.exp(k)
        mat[1, 0] = np.cos(self.a)*np.sin(self.a)*np.exp(1j*self.b)*np.exp(k)
        mat[1, 1] = 1-((np.cos(self.a)*np.exp(k))**2)
        return mat

    def exp_value(self, matrix, state_vector):
        """Compute the expectation value of a Pauli matrix.

        Args:
            matrix: matrix to compute the expectation value for.
            state_vector (numpy.ndarray): The quantum state vector.

        Returns:
            float: The expectation value.
        """
        expectation = np.vdot(state_vector, matrix @ state_vector)
        return expectation.real
        # exp = np.dot(state_vector, matrix)
        # return np.trace(exp)

    def derivative(self, func):
        # Define the variable
        t = symbols('t')
        # Define the function
        # func = t ** 2 + 3 * t + 1
        # Take the derivative with respect to t
        f_derivative = diff(func, t)
        print('Derviative: ', f_derivative)
        return f_derivative

    # Anticommutator --> {}
    def anticomm(self, matrix1, matrix2):
        return np.dot(matrix1, matrix2) + np.dot(matrix2, matrix1)

    # Commutator --> []
    def comm(self, matrix1, matrix2):
        return np.dot(matrix1, matrix2) - np.dot(matrix2, matrix1)

    # Implementation of Hamiltonian of the system
    # Eq. 5 in "Quantum Memristors with Quantum Computers"
    def hamiltonian(self):
        return (1/2)*self.h*self.w*(pauli_z2+2)

    # Implementation of master equation of the system
    # Eq. 10 in "Quantum Memristors with Quantum Computers"
    def master_eq_I(self, ts, p_I):
        return self.gamma(ts)*((np.dot(np.dot(pauli_low2, p_I), pauli_ris2))-(self.anticomm(np.dot(pauli_ris2, pauli_low2), p_I)))

    # Implementation of master equation of the system
    # Eq. 6 in "Quantum Memristors with Quantum Computers"
    def master_eq_2(self, ts, p_2):
        H = self.hamiltonian()
        return ((-1j/self.h)*self.comm(H, p_2))+self.gamma(ts)*((np.dot(np.dot(pauli_low2, p_2), pauli_ris2))-(self.anticomm(np.dot(pauli_ris2, pauli_low2), p_2)))

    def get_Schrödinger(self, t, p_I):
        # The function transforms the density matrix p_I at time t to the Schrödinger picture
        # i.e. it returns the corresponding p_2 at time t
        return np.exp((-1j*t)/(self.h*self.hamiltonian()))*p_I*np.exp((1j*t)/(self.h*self.hamiltonian()))


if __name__ == '__main__':
    # Time-steps
    eps = 0.1
    tmax = 100.1
    t = np.arange(0, tmax, eps)

    # Simulation parameters
    a = np.pi / 4
    b = np.pi / 5
    m = 1
    h = 1
    w = 1
    y0 = 0.4

    mem = memristor(y0, w, h, m, a, b)

    t_plt = time_plot.Tplot()

    pure_state = np.array([np.cos(a), np.sin(a) * np.exp(1j * b)], dtype=complex)

    for i in range(len(t)-1):
        print('Timestep: ', t[i])

        k_val = mem.k(t[i], t[i+1])
        print('K: ', k_val)

        pI_mat = mem.get_density_mat(k_val)
        print('p_I Density Matrix: ')
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in pI_mat]))

        # Master Equation
        pI_me = mem.master_eq_I(t[i], pI_mat)
        print('p_I Master Equation: ')
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in pI_me]))

        p2_mat = mem.get_Schrödinger(t[i], pI_mat)
        print('p_2 Schrödinger picture: ')
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in p2_mat]))

        # Master Equation
        p2_me = mem.master_eq_2(t[i], p2_mat)
        print('p_2 Master Equation: ')
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in p2_me]))

        # exp_pI = mem.exp_value(pI_me, pure_state)
        # exp_p2 = mem.exp_value(p2_me, pure_state)
        # Note: values of both master equations are the same because it's the same calculations with different equations
        exp_pI = np.trace(pI_me)
        exp_p2 = np.trace(p2_me)
        print('Expectation Value pI density matrix: ', np.trace(pI_mat))
        print('Expectation Value pI master equation: ', exp_pI)
        print('Expectation Value p2 density matrix: ', np.trace(p2_mat))
        print('Expectation Value p2 master equation: ', exp_p2)
        print()

        t_plt.update(t[i], exp_pI, exp_p2)
