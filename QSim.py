from QHH import QHH
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    qhh = QHH()
    
    eps = 0.0001
    tmax= 10
    ts = np.arange(0, tmax, eps)
    Zk = np.zeros(len(ts))
    ZNa = np.zeros(len(ts))

    
    ### First set of parameters
    # Cc=10.0**(-3)
    # Cg=10.0**(-3)
    # Cr=10.0**(-3)
    # w=10.0
    # I0=0.35
    # ZL=6000.0
    # Zout=55.0
    # n0=0.4
    # m0=0.1
    # h0=0.6
    # Zk[0] = 2000.0
    # ZNa[0] = 30000.0
       
    ### Spike try parameters
    Cc=10**(-6)
    Cg=10**(-6)
    Cr=10**(-6)
    # w=10**1
    # I0=1
    Zout=50
    n0=0.4
    m0=0.6
    h0=0.2
    ZL= 1 / (3*10**(-4))
    Zk[0] = 1 / (1.33*(n0**4))
    ZNa[0] = 1 / (0.17*(m0**3)*h0)

    
    w = np.full((len(ts)),10)
    I0 = np.full((len(ts)),1)
    I0[:int(len(ts)/4)] = 0
    w[:int(len(ts)/4)] = 0
    I = np.multiply(I0, np.sin(np.multiply(w, ts)))
    
    Vm = np.zeros(len(ts))
    Vm[0] = qhh.V(Zk[0], ZNa[0], ZL, Zout, I0[0], w[0], ts[0], Cc, Cr)
          
    for i in range(len(ts)-1):
        t = ts[i]

        k1 = qhh.k(t, eps, Zk[i], ZNa[i], ZL, Zout, I0[i], w[i], Cc, Cr, n0)
        q1 = qhh.q(t, eps, Zk[i], ZNa[i], ZL, Zout, I0[i], w[i], Cc, Cr, m0, h0)
                
        k2 = qhh.k(t+0.5*eps, eps, Zk[i]+0.5*k1, ZNa[i]+0.5*q1, ZL, Zout, I0[i], w[i], Cc, Cr, n0)
        q2 = qhh.q(t+0.5*eps, eps, Zk[i]+0.5*k1, ZNa[i]+0.5*q1, ZL, Zout, I0[i], w[i], Cc, Cr, m0, h0)

        k3 = qhh.k(t+0.5*eps, eps, Zk[i]+0.5*k2, ZNa[i]+0.5*q2, ZL, Zout, I0[i], w[i], Cc, Cr, n0)
        q3 = qhh.q(t+0.5*eps, eps, Zk[i]+0.5*k2, ZNa[i]+0.5*q2, ZL, Zout, I0[i], w[i], Cc, Cr, m0, h0)

        k4 = qhh.k(t+eps, eps, Zk[i]+k3, ZNa[i]+q3, ZL, Zout, I0[i], w[i], Cc, Cr, n0)
        q4 = qhh.q(t+eps, eps, Zk[i]+k3, ZNa[i]+q3, ZL, Zout, I0[i], w[i], Cc, Cr, m0, h0)
        
        Zk[i+1] = Zk[i] + (1/6)*(k1+2*k2+2*k3+k4); 
        ZNa[i+1] = ZNa[i] + (1/6)*(q1+2*q2+2*q3+q4);
        
        Vm[i+1] = qhh.V(Zk[i+1], ZNa[i+1], ZL, Zout, I0[i], w[i], ts[i+1], Cc, Cr)
        
    Gk = 1.0 / Zk 
    GNa = 1.0 / ZNa
    
    plt.subplot(2,2,1)
    plt.plot(ts, I, 'b')
    plt.title("I")
    
    plt.subplot(2,2,2)
    plt.plot(ts, Vm, 'r')
    plt.title("V")
    
    plt.subplot(2,2,3)
    plt.plot(ts, Gk, 'g')
    plt.title("Gk")
    
    plt.subplot(2,2,4)
    plt.plot(ts, GNa, 'y')
    plt.title("GNa")
    
    plt.tight_layout()
    plt.show()