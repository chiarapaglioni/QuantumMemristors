from QHH_1 import QHH_1
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def hh_model_quantized(t, y, I, Vc, k1, k2, k3, k4):
    """
    Implementation of the Quantized Single-Ion-Channel Hodgkin-Huxley Model for Quantum Neurons.

    Args:
    - t: float, time point at which to evaluate the model
    - y: list of floats, the current values of the variables in the model
    - I: float, external current input
    - Vc: float, capacitance of the cell
    - k1, k2, k3, k4: float, model parameters

    Returns:
    - dydt: list of floats, the derivative of each variable at time t
    """
    # Define variables
    V, m, h, n = y

    # Define differential equations
    dVdt = (I - k1 * m ** 3 * h * (V - k2)) / Vc
    dmdt = k3 * (1 - m) * np.exp(-(V - k4) / 10) - k3 * m * np.exp((V - k4) / 10)
    dhdt = k3 * (1 - h) * np.exp(-(V - k4) / 10) - k3 * h * np.exp((V - k4) / 10)
    dndt = k3 * (1 - n) * np.exp(-(V - k4) / 80) - k3 * n * np.exp((V - k4) / 80)

    # Return derivatives
    dydt = [dVdt, dmdt, dhdt, dndt]
    return dydt

"""
    Simulation of Quantized three-ion-channel Hodgkin-Huxley model
"""
if __name__ == "__main__":
    qhh = QHH_1()
    
    # Define initial conditions and parameters
    y0 = [-65, 0.05, 0.6, 0.32]
    I = 10
    Vc = 1
    k1 = 120
    k2 = -50
    k3 = 0.1
    k4 = -77

    # Define time points to evaluate the model
    t = np.linspace(0, 100, 1000)

    temp = hh_model_quantized(t, y0, I, Vc, k1, k2, k3, k4)
    print(temp)

    # Solve the differential equations
    sol = odeint(hh_model_quantized, y0, t, args=(I, Vc, k1, k2, k3, k4))

    print(sol)

    plt.plot(t, sol[:, 0], label='V')
    plt.plot(t, sol[:, 1], label='m')
    plt.plot(t, sol[:, 2], label='h')
    plt.plot(t, sol[:, 3], label='n')
    plt.legend()
    plt.xlabel('Time (ms)')
    plt.ylabel('State variable value')
    plt.show()
