from q_memristor.numerical.operators import *
from num_memristor import memristor
from q_memristor.plots import iv_plot, time_plot
import numpy as np


if __name__ == '__main__':
    # Time-steps
    eps = 0.1
    tmax = 100.1
    t = np.arange(0, tmax, eps)

    # Simulation parameters
    a = np.pi/4
    b = np.pi/5
    m = 1
    h = 1
    w = 1
    y0 = 0.4
    amplitude = 1
    ang_freq = w                  # angular frequency = w

    mem = memristor(y0, w, h, m, a, b)

    angle = np.arccos(np.exp(mem.k1(0)))    # phase angle

    # Plots types:
    #   - 'iv' = IV plot
    #   - 't' = time plot
    plot_type = 't'

    pure_state = np.array([np.cos(a), np.sin(a) * np.exp(1j * b)], dtype=complex)

    # V = sinusoidal time dependence
    # Updated by the following function:
    # amplitude * np.sin(ang_freq * t[0] + angle))
    V = []
    I = []

    # Initialize V_0 and I_0
    V.append(-(1 / 2) * np.sqrt((m * h * w) / 2) * mem.exp_value(pauli_y2, pure_state))
    # V.append(amplitude * np.sin(ang_freq * t[0] + angle))
    I.append(mem.gamma(t[0]) * V[0])
    print('V: ', V[0], ' at time: ', 0.0)
    print('I: ', I[0], ' at time: ', 0.0)

    iv_plt = iv_plot.IVplot(V[0], I[0])
    t_plt = time_plot.Tplot()

    for i in range(1, len(t)):
        # I is evaluated correcty however there is an error with the update of the voltage V at each iteration
        # angle = np.arccos(np.exp(mem.k1(t[i])))
        # v_val = amplitude * np.sin(ang_freq * t[i] + angle)
        # V.append(V[0]*np.cos(w*t[i]))
        v_val = V[0]*np.sin(w*t[i])
        V.append(v_val)
        I.append(mem.gamma(t[i]) * V[i])
        iv_plt.update(V[i], I[i])
        t_plt.update(t[i], V[i], I[i])
        print('V: ', V[i], ' at time: ', t[i])
        print('I: ', I[i], ' at time: ', t[i])

    iv_plt.save_plot()
    t_plt.save_plot()

    V = V/V[0]
    I = I/I[0]

    print(V)
    print(I)
