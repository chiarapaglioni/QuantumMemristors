import numpy as np


class QHH_1:
    """
    Implementation of Quantized single-ion-channel Hodgkin-Huxley model
    (Only potassium channel (K) is considered)
    """

    def __init__(self):
        pass

    # Z(t) = Z_min * n(t)^-4
    def Z(self, Zk):
        return Zk

    def alpha_n(self, V):
        return 0.02 * (V + 45.7) / (1 - np.exp(-0.1 * (V + 45.7)))

    def beta_n(self, V):
        return 0.25 * np.exp(-0.0125 * (V + 55.7))

    def alpha_m(self, V):
        return 0.182 * (V + 35) / (1 - np.exp(-0.1 * (V + 35)))

    def beta_m(self, V):
        return 0.124 * np.exp(-0.0125 * (V + 35))

    def alpha_h(self, V):
        return 0.25 * np.exp(-0.05 * (V + 55.7))

    def beta_h(self, V):
        return 4 * np.exp(-0.0556 * (V + 55.7))

    def n(self, V, n0, t, alpha_n=None, beta_n=None):
        if alpha_n == None:
            alpha_n = self.alpha_n(V)
            beta_n = self.beta_n(V)
        return alpha_n/(alpha_n+beta_n) - ((alpha_n/(alpha_n+beta_n))-n0)*np.exp(-(alpha_n+beta_n)*t)

    def m(self, V, m0, t, alpha_m=None, beta_m=None):
        if alpha_m == None:
            alpha_m = self.alpha_m(V)
            beta_m = self.beta_m(V)
        return alpha_m/(alpha_m+beta_m) - ((alpha_m/(alpha_m+beta_m))-m0)*np.exp(-(alpha_m+beta_m)*t)

    def h(self, V, h0, t, alpha_h=None, beta_h=None):
        if alpha_h == None:
            alpha_h = self.alpha_h(V)
            beta_h = self.beta_h(V)
        return alpha_h/(alpha_h+beta_h) - ((alpha_h/(alpha_h+beta_h))-h0)*np.exp(-(alpha_h+beta_h)*t)

    # Voltage of the system
    # Function 37
    def V(self, Z, I0, w, t, Cc):
        f = Z*I0*(np.sin(w*t)-(Cc*w*Z*np.cos(w*t))) / (1.0+(Cc**2.0)*(w**2)*(Z**2))
        return f

    # Update of the system
    # Formula of Z(t) --> number 27
    def k(self, t, eps, Zk, n0):
        V = self.V(Zk, n0)
        alpha_n = self.alpha_n(V)
        beta_n = self.beta_n(V)
        n = self.n(V, n0, t, alpha_n, beta_n)
        f = eps*(-4*Zk * (alpha_n / n - (alpha_n + beta_n)))
        return f

    # Update of the system
    # TODO: implement update rule R from photonic quantum memristor
    def r(self):
        pass

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

    