import numpy as np


class QHH_1:
    """
    Implementation of Quantized single-ion-channel Hodgkin-Huxley model
    (Only potassium channel (K) is considered)
    """

    def __init__(self):
        pass

    # Z(t) = Z_min * n(t)^-4
    def Zmin(self, Zk, n):
        return Zk / n**-4

    def alpha_n(self, V):
        return 0.01*(V+55) / (1-np.exp(-(V+55)/10))

    def beta_n(self, V):
        return 0.125*np.exp(-(V+65)/80)

    # Voltage of the system
    # Function 37
    def V(self, Z, I0, w, t, Cc):
        f = Z*I0*(np.sin(w*t)-(Cc*w*Z*np.cos(w*t))) / (1.0+(Cc**2.0)*(w**2)*(Z**2))
        return f

    # Update of the system
    # Formula of Z(t) --> number 27
    def k(self, t, eps, Zk, I0, w, Cc, n0):
        V = self.V(Zk, I0, w, t, Cc)
        alpha_n = self.alpha_n(V)
        beta_n = self.beta_n(V)
        Z_min = self.Zmin(Zk, n0)
        f = eps*(-4*Z_min * (Zk/Z_min)**(5/4) * alpha_n + 4*Zk*(alpha_n+beta_n))
        return f

    # Update of the system
    # TODO: implement update rule R from photonic quantum memristor
    def r(self):
        pass

    