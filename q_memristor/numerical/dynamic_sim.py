from q_memristor.numerical.operators import *
from q_memristor.plots import iv_plot, time_plot
from num_memristor import memristor
import numpy as np

"""
    Dynamic Simulation of Single Quantum Memristor based on the article "Quantum Memristors with Quantum 
    Computers" from Y.-M. Guo, F. Albarr ́an-Arriagada, H. Alaeian, E. Solano, and G. Alvarado Barrios. 

    Author: Chiara Paglioni
    Link to Article: https://link.aps.org/doi/10.1103/PhysRevApplied.18.024082  
    
    Experiment: 
    Here the dynamical numerical simulation of the single memristive dynamics is implemented. 
"""


if __name__ == '__main__':
    # Time-steps
    eps = 0.1
    tmax = 1.1
    t = np.arange(0, tmax, eps)

    # Simulation parameters
    a = np.pi / 4
    b = np.pi / 5
    m = 1
    h = 1
    w = 1
    y0 = 0.4
    amplitude = 1

    pure_state = np.array([np.cos(a), np.sin(a) * np.exp(1j * b)], dtype=complex)

    mem = memristor(y0, w, h, m, a, b)

    k = []
    density_states = []
    density_states2 = []
    schrodinger_states = []

    V = []
    I = []

    t_plt = time_plot.Tplot()

    # Initialize density matrix at time 0
    k_val0 = mem.k1(0)
    density_mat0 = mem.get_density_mat(k_val0)
    schrodinger_mat0 = mem.get_Schrodinger(0, density_mat0)
    density_states.append(density_mat0)
    schrodinger_states.append(schrodinger_mat0)

    for i in range(1, len(t)-1):
        k_val = mem.k(t[i], t[i+1])

        density_mat = mem.get_E0(t[i], t[i+1]) @ density_states[0] @ mem.adjoint(mem.get_E0(t[i], t[i+1])) + mem.get_E1(t[i], t[i+1]) @ density_states[0] @ mem.adjoint(mem.get_E1(t[i], t[i+1]))
        pI_me = mem.master_eq_I(t[i], density_mat)
        schrodinger_mat = mem.get_Schrodinger(t[i], density_mat)
        p2_me = mem.master_eq_2(t[i], schrodinger_mat)

        density_states.append(density_mat)
        schrodinger_states.append(schrodinger_mat)

        k.append(k_val)

        print('Time: ', t[i])
        print('K: ', k_val)
        print('Density Matrix: ')
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in density_mat]))
        print('Schrodinger Picture: ')
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in schrodinger_mat]))
        print('p_I Master Equation: ')
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in pI_me]))
        print('p_2 Master Equation: ')
        print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in p2_me]))

        exp_valueY = np.trace(pauli_y @ schrodinger_mat)
        exp_valueX = np.trace(pauli_x @ schrodinger_mat)
        exp_valueZ = np.trace(pauli_z @ schrodinger_mat)

        print("Expectation value Pauli Y:", exp_valueY)
        print("Expectation value Pauli X:", exp_valueX)
        print("Expectation value Pauli Z:", exp_valueZ)

        vol_val = -(1 / 2) * np.sqrt((m * h * w) / 2) * exp_valueY
        # curr_val = np.sqrt((m * h * w) / 2) * exp_valueY - np.sqrt((m * w) / (2*h)) * exp_valueX
        curr_val = mem.gamma(t[i]) * vol_val

        V.append(vol_val)
        I.append(curr_val)
        print('V: ', V[i], ' at time: ', t[i])
        print('I: ', I[i], ' at time: ', t[i])
        print()

        t_plt.update(t[i], V[i], I[i])

    t_plt.save_plot()
