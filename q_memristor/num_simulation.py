import numpy as np
from scipy.integrate import quad


def gamma(ts):
    y0 = 0.4
    w = 1
    return y0 * (1 - np.sin(np.cos(w * ts)))


def k(ts_next, ts):
    integral_result, _ = quad(gamma, ts, ts_next)
    return -integral_result / 2


def p(a, b, k):
    mat = np.empty((2, 2), dtype='complex_')
    mat[0, 0] = (np.cos(a)*np.exp(k))**2
    mat[0, 1] = np.cos(a)*np.sin(a)*np.exp(-1j*b)*np.exp(k)
    mat[1, 0] = np.cos(a)*np.sin(a)*np.exp(1j*b)*np.exp(k)
    mat[1, 1] = 1-((np.cos(a)*np.exp(k))**2)
    return mat


# Anticommutator of two quantum operands
def anticomm(matrix1, matrix2):
    return np.dot(matrix1, matrix2) + np.dot(matrix2, matrix1)


# Commutator of two quantum operands
def comm(matrix1, matrix2):
    return np.dot(matrix1, matrix2) - np.dot(matrix2, matrix1)


def hamiltonian(h, w, p_z):
    p_temp = np.array(p_z)+2
    return (h*w*p_temp)/2


if __name__ == '__main__':
    eps = 0.1
    tmax = 1.2
    t = np.arange(0, tmax, eps)
    a = np.pi/4
    b = np.pi/5
    h = 1
    w = 1

    pauli_low = [[0, 1],
                 [0, 0]]
    pauli_ris = [[0, 0],
                 [1, 0]]
    pauli_z = [[1, 0],
               [0, -1]]

    for i in range(len(t)-1):
        g = gamma(t[i])
        k_val = k(t[i+1], t[i])
        p_mat = p(a, b, k_val)
        h2 = hamiltonian(h, w, pauli_z)
        mat3 = comm(h2, p_mat)
        mat1 = np.dot(np.dot(pauli_low, p_mat), pauli_ris)
        mat2 = anticomm(np.dot(pauli_low, pauli_low), p_mat)
        out = -((1j*mat3)/h) + (g * (mat1 - (mat2/2)))

        print('Timestep: ', t[i])
        print('Gamma: ', g)
        print('K: ', k_val)
        print('p_I: ')
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in p_mat]))
        print('H2: ')
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in h2]))
        print('matrix1: ')
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in mat1]))
        print('matrix2: ')
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in mat2]))
        print('output: ')
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in out]))
        print()

