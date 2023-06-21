import numpy as np

from q_memristor.numerical.num_memristor import memristor
from q_memristor.plots import time_plot

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
    