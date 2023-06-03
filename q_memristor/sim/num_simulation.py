import numpy as np
from scipy.integrate import quad
from sympy import symbols, diff
from q_memristor.plots import iv_plot, time_plot

# Note that --> gamma has a sinusoidal time dependance
# Sinusoidal time dependence refers to a quantity or phenomenon that varies with time in a sinusoidal or harmonic manner
# It is characterized by a periodic oscillation that follows a sine or cosine function.
def gamma(ts):
    y0 = 0.4
    w = 1
    return y0 * (1 - np.sin(np.cos(w * ts)))


def gamma2(y0, w, ts):
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


def exp_value(pauli_matrix, state_vector):
    """Compute the expectation value of a Pauli matrix.

    Args:
        pauli_matrix (str): The Pauli matrix to compute the expectation value for.
                            Can be 'X', 'Y', or 'Z'.
        state_vector (numpy.ndarray): The quantum state vector.

    Returns:
        float: The expectation value.
    """
    # Define the corresponding Pauli matrix operators
    pauli_operators = {
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex)
    }

    # Select the appropriate Pauli matrix operator
    if pauli_matrix not in pauli_operators:
        raise ValueError("Invalid Pauli matrix. Must be 'X', 'Y', or 'Z'.")
    pauli_operator = pauli_operators[pauli_matrix]

    # Compute the expectation value
    expectation = np.vdot(state_vector, pauli_operator @ state_vector)
    return expectation.real


def derivative(func):
    # Define the variable
    t = symbols('t')
    # Define the function
    # func = t ** 2 + 3 * t + 1
    # Take the derivative with respect to t
    f_derivative = diff(func, t)
    print('Derviative: ', f_derivative)
    return f_derivative


# Anticommutator
def anticomm(matrix1, matrix2):
    return np.dot(matrix1, matrix2) + np.dot(matrix2, matrix1)


# Commutator
def comm(matrix1, matrix2):
    return np.dot(matrix1, matrix2) - np.dot(matrix2, matrix1)


def hamiltonian(h, w, p_z):
    p_temp = np.array(p_z)+2
    return (h*w*p_temp)/2


if __name__ == '__main__':
    eps = 0.1
    tmax = 100.1
    t = np.arange(0, tmax, eps)
    a = np.pi/4
    b = np.pi/5
    m = 1
    h = 1
    w = 1
    y0 = 0.4

    pauli_low = [[0, 1],
                 [0, 0]]
    pauli_ris = [[0, 0],
                 [1, 0]]
    pauli_z = [[1, 0],
               [0, -1]]

    pure_state = np.array([np.cos(a), np.sin(a) * np.exp(1j * b)], dtype=complex)

    iv_plt = iv_plot.IV_plot()
    t_plt = time_plot.t_plot(len(t))

    V = []
    I = []

    # Initialize V_0 and I_0
    V.append(-(1 / 2) * np.sqrt((m * h * w) / 2) * exp_value('Y', pure_state))
    I.append(gamma2(y0, w, t[0]) * V[0])
    print('V: ', V[0], ' at time: ', 0.0)
    print('I: ', I[0], ' at time: ', 0.0)

    for i in range(1, len(t)):
        V.append(V[0]*np.cos(w*t[i]))
        I.append(gamma2(y0, w, t[i]) * V[i])
        iv_plt.update(V[i], I[i])
        t_plt.update(t[i], V[i], I[i])
        print('V: ', V[i], ' at time: ', t[i])
        print('I: ', I[i], ' at time: ', t[i])

    V = V/V[0]
    I = I/I[0]

    print(V)
    print(I)

    # for i in range(len(t)-1):
    #     g = gamma(t[i])
    #     k_val = k(t[i+1], t[i])
    #     p_mat = p(a, b, k_val)
    #     h2 = hamiltonian(h, w, pauli_z)
    #     mat3 = comm(h2, p_mat)
    #     mat1 = np.dot(np.dot(pauli_low, p_mat), pauli_ris)
    #     mat2 = anticomm(np.dot(pauli_low, pauli_low), p_mat)
    #     out = -((1j*mat3)/h) + (g * (mat1 - (mat2/2)))
    #
    #     print('Timestep: ', t[i])
    #     print('Gamma: ', g)
    #     print('K: ', k_val)
    #     print('p_I: ')
    #     print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in p_mat]))
    #     print('H2: ')
    #     print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in h2]))
    #     print('Commutator: ')
    #     print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in mat3]))
    #     print('matrix1: ')
    #     print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in mat1]))
    #     print('matrix2: ')
    #     print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in mat2]))
    #     print('output: ')
    #     print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in out]))
    #     print(np.trace(out))
    #     print()

    # plt.plot(t, V, color='red', label='Voltage')
    # plt.plot(t, I, color='blue', label='Current')

    # Customize the plot (optional)
    # plt.title('Sine Wave')  # Set the plot title
    # plt.xlabel('x')  # Set the x-axis label
    # plt.ylabel('y')  # Set the y-axis label

    # Display the plot
    # plt.show()
