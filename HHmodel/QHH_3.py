import numpy as np


class QHH_3:
    """
    Implementation of Quantized three-ion-channel Hodgkin-Huxley model
    (Sodium (Na), Potassium (K) and Leakage channel are considered)
    """

    def __init__(self):
        pass

    # Function used to compute theta
    def T(self, Zk, ZNa, ZL):
        T = np.sqrt(Zk*ZL + ZNa*ZL + Zk*ZNa) / (np.sqrt(Zk*ZL) + np.sqrt(ZNa*ZL) + np.sqrt(Zk*ZNa))
        return T

    def Z(self, Zk, ZNa, ZL):
        return Zk*ZNa*ZL / (Zk*ZL + ZNa*ZL + Zk*ZNa)

    def alpha_n(self, V):
        return 0.01*(V+55) / (1-np.exp(-(V+55)/10))

    def beta_n(self, V):
        return 0.125*np.exp(-(V+65)/80)

    def n(self, V, n0, t, alpha_n=None, beta_n=None):
        if alpha_n == None:
            alpha_n = self.alpha_n(V)
            beta_n = self.beta_n(V)
        return alpha_n/(alpha_n+beta_n) - ((alpha_n/(alpha_n+beta_n))-n0)*np.exp(-(alpha_n+beta_n)*t)

    def alpha_m(self, V):
        return 0.1*(V+40)/(1-np.exp(-(V+40)/10))

    def beta_m(self, V):
        return 4*np.exp(-(V+65)/18)

    def m(self, V, m0, t, alpha_m=None, beta_m=None):
        if alpha_m == None:
            alpha_m = self.alpha_m(V)
            beta_m = self.beta_m(V)
        return alpha_m/(alpha_m+beta_m) - ((alpha_m/(alpha_m+beta_m))-m0)*np.exp(-(alpha_m+beta_m)*t);

    def alpha_h(self, V):
        return 0.07*np.exp(-(V+65)/20)

    def beta_h(self, V):
        return 1/(1+np.exp(-(V+35)/10))

    def h(self, V, h0, t, alpha_h=None, beta_h=None):
        if alpha_h == None:
            alpha_h = self.alpha_h(V)
            beta_h = self.beta_h(V)
        return alpha_h/(alpha_h+beta_h) - ((alpha_h/(alpha_h+beta_h))-h0)*np.exp(-(alpha_h+beta_h)*t)

    # Voltage of the system
    def V(self, Zk, ZNa, ZL, Zout, I0, w, t, Cc, Cr):
        Z = self.Z(Zk, ZNa, ZL)
        T = self.T(Zk, ZNa, ZL)
        f = Z*T*I0*(-w*Z*T*(Cc+Cr+Cc*(Cr**2.0)*(w**2.0)*(Zout**2.0))*np.cos(w*t)+(1.0+(Cr**2.0)*(w**2.0)*Zout*(Z*T+Zout))*np.sin(w*t))/(1.0+(w**2.0)*((((Cc+Cr)*Z*T)**2.0)+2.0*(Cr**2.0)*Z*T*Zout+((Cr*Zout)**2.0)*(1.0+((Cc*w*Z*T)**2.0))))
        return f

    # Update of the system
    # Formula of Zk
    def k(self, t, eps, Zk, ZNa, ZL, Zout, I0, w, Cc, Cr, n0):
        V = self.V(Zk, ZNa, ZL, Zout, I0, w, t, Cc, Cr)
        alpha_n = self.alpha_n(V)
        beta_n = self.beta_n(V)
        n = self.n(V, n0, t, alpha_n, beta_n)
        f = eps*(-4*Zk * (alpha_n / n - (alpha_n + beta_n)))
        return f

    # Update of the system
    # Formula of ZNa
    def q(self, t, eps, Zk, ZNa, ZL, Zout, I0, w, Cc, Cr, m0, h0):
        V = self.V(Zk, ZNa, ZL, Zout, I0, w, t, Cc, Cr)
        alpha_m = self.alpha_m(V)
        beta_m = self.beta_m(V)
        m = self.m(V, m0, t, alpha_m, beta_m)

        alpha_h = self.alpha_h(V)
        beta_h = self.beta_h(V)
        h = self.h(V, h0, t, alpha_h, beta_h)

        f = eps*(-ZNa * (3*((alpha_m / m) - (alpha_m + beta_m)) + (alpha_h / h) - (alpha_h + beta_h)))
        return f
    