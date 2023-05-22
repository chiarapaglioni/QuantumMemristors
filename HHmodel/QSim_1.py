from QHH_1 import QHH_1
import numpy as np
import matplotlib.pyplot as plt


"""
    Simulation of Quantized three-ion-channel Hodgkin-Huxley model
"""
if __name__ == "__main__":
    qhh = QHH_1()

    eps = 0.0001
    tmax = 10
    ts = np.arange(0, tmax, eps)
    Zk = np.zeros(len(ts))

    # Spike try parameters
    Cc = 10 ** (-6)

    # Impedance of the outgoing transmission line
    Zout = 50

    # Activation variable value
    n0 = 0.4

    # Initial update of the system
    # Chloride channel = constant = 1 / GK
    # GK = max potassium conductance * m0^4
    Zk[0] = 1 / (1.33 * (n0 ** 4))

    w = np.full((len(ts)), 10)
    I0 = np.full((len(ts)), 1)
    I0[:int(len(ts) / 4)] = 0
    w[:int(len(ts) / 4)] = 0

    # Input current of the system
    I = np.multiply(I0, np.sin(np.multiply(w, ts)))

    # Voltage
    Vm = np.zeros(len(ts))
    Vm[0] = qhh.V(Zk[0], I0[0], w[0], ts[0], Cc)

    # Update of the system
    # TODO: Fix implementation of QHH only with Potassium K Ion
    for i in range(len(ts) - 1):
        t = ts[i]

        k1 = qhh.k(t, eps, Zk[i], I0[i], w[i], Cc, n0)

        k2 = qhh.k(t + 0.5 * eps, eps, Zk[i] + 0.5 * k1, I0[i], w[i], Cc, n0)

        k3 = qhh.k(t + 0.5 * eps, eps, Zk[i] + 0.5 * k2, I0[i], w[i], Cc, n0)

        k4 = qhh.k(t + eps, eps, Zk[i] + k3, I0[i], w[i], Cc, n0)

        # Update voltage of the potassium channel
        Zk[i + 1] = Zk[i] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Update voltage of the system
        Vm[i + 1] = qhh.V(Zk[i + 1], I0[i], w[i], ts[i + 1], Cc)

    Gk = 1.0 / Zk

    plt.subplot(3, 1, 1)
    plt.plot(ts, I, 'b')
    plt.title("Input Current")

    plt.subplot(3, 1, 2)
    plt.plot(ts, Vm, 'r')
    plt.title("Voltage")

    plt.subplot(3, 1, 3)
    plt.plot(ts, Gk, 'g')
    plt.title("Potassium Conductance Gk")

    plt.tight_layout()
    plt.show()

