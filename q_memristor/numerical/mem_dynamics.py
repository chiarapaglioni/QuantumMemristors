from q_memristor.numerical.operators import *
from q_memristor.plots import iv_plot, time_plot
from num_memristor import memristor
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
    tmax = 15.1
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
    V.append(-(1 / 2) * np.sqrt((m * h * w) / 2) * mem.exp_value(pauli_y, pure_state))
    I.append(mem.gamma(t[0]) * V[0])
    print('V: ', V[0], ' at time: ', 0.0)
    print('I: ', I[0], ' at time: ', 0.0)

    iv_plt = iv_plot.IVplot(V[0], I[0])
    t_plt = time_plot.Tplot()

    # V = sinusoidal time dependence
    # Hysterisis is constant because of V0 however V[0] should decrease in time
    # function of gamma and t to make V[0] decrease such that it is smaller than the previous V[0]
    for i in range(1, len(t)):
        v_val = 0

        # angle = np.arccos(np.exp(mem.k1(t[i])))
        # v_val = amplitude * np.sin(w * t[i] + angle)
        # v_val = f(V[0], y0, t[i]) * np.sin(w * t[i])
        # v_val = f(V[0], mem.gamma(t[i]), t[i]) * np.sin(w * t[i])

        if dynamic:
            v_val = f(V[0], y0, t[i]) * np.sin(w * t[i])
        else:
            v_val = V[0] * np.sin(w*t[i])

        i_val = mem.gamma(t[i]) * v_val

        V.append(v_val)
        I.append(i_val)

        iv_plt.update(V[i], I[i])
        t_plt.update(t[i], V[i], I[i])
        print('V: ', V[i], ' at time: ', t[i])
        print('I: ', I[i], ' at time: ', t[i])
        print()

    iv_plt.save_plot()
    t_plt.save_plot()

    V = V/V[0]
    I = I/I[0]

    print('Normalized Voltage: ')
    print(V)
    print('Normalized Current: ')
    print(I)
