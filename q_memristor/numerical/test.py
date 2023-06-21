from mpmath import quad
import numpy as np


def decay_rate(t, a):
    return -2 * np.real(c1(t, a), c1(t, a))


def k(t):
    result, _ = quad(decay_rate, 0, t)
    return -result / 2


def c1(t, a):
    return np.cos(a) * np.exp(k(t))


def get_density_mat(a, b, k):
    mat = np.empty((2, 2), dtype=complex)
    mat[0, 0] = (np.cos(a)*np.exp(k))**2
    mat[0, 1] = np.cos(a)*np.sin(a)*np.exp(-1j*b)*np.exp(k)
    mat[1, 0] = np.cos(a)*np.sin(a)*np.exp(1j*b)*np.exp(k)
    mat[1, 1] = 1-((np.cos(a)*np.exp(k))**2)
    return mat


def hamiltonian(h, w, pauli_z):
    return (1/2) * h * w * (pauli_z + 2)


if __name__ == '__main__':
    # Parameters at t = 0
    a = np.pi/8
    b = np.pi/5
    y0 = 0.4
    k_val0 = -0.02/2
    test = -2 * np.real((-np.sin(a))/(np.cos(a)))

    h = 1
    w = 1
    m = 1
    pauli_z = np.array([[1, 0], [0, -1]])
    pauli_y = np.array([[0, -1j], [1j, 0]])

    print((-np.sin(a))/(np.cos(a)))
    print(test)
    print((np.cos(a)*y0)/-2)
    print(-np.sin(a))
    print(k_val0)
    print()

    pI_mat = get_density_mat(a, b, k_val0)
    ham = hamiltonian(h, w, pauli_z)

    # At t = 0 the Schrodinger picture is equal to the density mat

    exp_valueY = np.trace(pauli_y @ pI_mat)
    v0_val = -(1 / 2) * np.sqrt((m * h * w) / 2) * exp_valueY

    print(pI_mat)
    print()
    print(ham)
    print()
    print(exp_valueY)
    print(v0_val)
