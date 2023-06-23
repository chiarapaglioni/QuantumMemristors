from q_memristor.numerical.operators import *
from q_memristor.plots import iv_plot, time_plot
from num_memristor import memristor
from q_memristor.circuits.csv_generator import csv_gen
import numpy as np

"""
    Simulation of Generalized Quantum Memristive Dynamics for Sinusoidal Time Dependent Input

    Author: Chiara Paglioni
    Link to Articles: 
        - https://link.aps.org/doi/10.1103/PhysRevApplied.18.024082 
        - https://doi.org/10.1038/srep42044
     
"""


def f(v0, gamma_t, ts):
    """
        Function used to update the voltage of the system based on:
        - initial voltage v0
        - decay rate: gamma_t
        - time-step: ts
    """
    # return v0 * gamma_t
    return v0 + (gamma_t * ts)


if __name__ == '__main__':
    # Time-steps
    eps = 0.1
    tmax = 10.1
    t = np.arange(0, tmax, eps)

    # Simulation parameters
    a = np.pi/4
    b = np.pi/5
    m = 1
    h = 1
    w = 1
    y0 = 0.2

    # Update of V0 is not based on dynamic values of gamma
    dynamic = True

    mem = memristor(y0, w, h, m, a, b)

    angle = np.arccos(np.exp(mem.k1(0)))

    pure_state = np.array([np.cos(a), np.sin(a) * np.exp(1j * b)], dtype=complex)

    V = []
    I = []

    # Initialize V0 and I0 through the memristive equations derived from the master equation
    V0 = -(1 / 2) * np.sqrt((m * h * w) / 2) * mem.exp_value(pauli_y, pure_state)
    I0 = mem.gamma(t[0]) * V0
    print('V0: ', V0)
    print('I0: ', I0)

    v_val = 0

    # Compute initial values of voltage and current based on V0 and I0
    if dynamic:
        v_val = f(V0, y0, t[0]) * np.sin(w * t[0])
    else:
        v_val = V0 * np.sin(w * t[0])

    i_val = mem.gamma(t[0]) * v_val

    V.append(v_val)
    I.append(i_val)

    iv_plt = iv_plot.IVplot(V[0], I[0], dynamic)
    t_plt = time_plot.Tplot(dynamic)

    print('V: ', V[0], ' at time: ', t[0])
    print('I: ', I[0], ' at time: ', t[0])

    # V = sinusoidal time dependence
    # Hysterisis is constant because of V0 however V[0] should decrease in time
    # function of gamma and t to make V[0] decrease such that it is smaller than the previous V[0]
    for i in range(1, len(t)):
        final_bool = False
        v_val = 0

        # angle = np.arccos(np.exp(mem.k1(t[i])))
        # v_val = amplitude * np.sin(w * t[i] + angle)
        # v_val = f(V[0], y0, t[i]) * np.sin(w * t[i])
        # v_val = f(V[0], mem.gamma(t[i]), t[i]) * np.sin(w * t[i])

        if dynamic:
            v_val = f(V0, y0, t[i]) * np.sin(w * t[i])
        else:
            v_val = V0 * np.sin(w*t[i])

        i_val = mem.gamma(t[i]) * v_val

        V.append(v_val)
        I.append(i_val)

        if t[i] == t[len(t)-1]:
            final_bool = True

        iv_plt.update(V[i], I[i], final_bool)
        t_plt.update(t[i], V[i], I[i])

        print('V: ', V[i], ' at time: ', t[i])
        print('I: ', I[i], ' at time: ', t[i])
        print()

    iv_plt.save_plot()
    t_plt.save_plot()

    if dynamic:
        csvGen = csv_gen('data/data_numMem_dyn.csv')
        csvGen.write_data(t, V, I)
    else:
        csvGen = csv_gen('data/data_numMem1_stat.csv')
        csvGen.write_data(t, V, I)

    V = V/V0
    I = I/I0

    print('Normalized Voltage: ')
    print(V)
    print('Normalized Current: ')
    print(I)
