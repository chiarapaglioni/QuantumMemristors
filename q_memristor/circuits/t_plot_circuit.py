import matplotlib.pyplot as plt

"""
    Basic implementation of time plot of Voltage (V) and Current (I) for quantum circuits

    Author: Chiara Paglioni
"""


class Tplot:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.ax.axhline(0, color='black', linewidth=0.5)
        self.xs = []
        self.voltages = []
        self.currents = []
        self.preferred_w = 1000
        self.preferred_h = 500
        self.v0 = -0.2033639102692511
        self.i0 = -0.012895632168239872
        # self.voltage_scale_factor = 100
        # self.current_scale_factor = 100000
        self.labels_displayed = False
        self.ax.set_xlabel('Time (t)')
        self.ax.set_ylabel('Current <I> & Voltage <V>')
        self.ax.set_title('Time Plot')

    def update(self, t, v, i):
        self.xs.append(t)
        # if t == 0:
        self.voltages.append(v)
        self.currents.append(i)
        # else:
        # self.voltages.append(v/self.v0)
        # self.currents.append(i/self.i0)
        # Uncomment the next line if we want to see only one dot at each update
        # self.ax.cla()
        self.ax.axhline(0, color='black', linewidth=0.5)
        self.ax.plot(self.xs, self.voltages, 'ro-', markersize=1, label='Voltage')
        self.ax.plot(self.xs, self.currents, 'bo-', markersize=1, label='Current')

        self.ax.set_xlim(0, t+0.1)
        self.ax.set_ylim(-0.3, 0.2)

        if not self.labels_displayed:
            self.ax.legend()
            self.labels_displayed = True

        plt.pause(0.001)

    def save_plot(self, plot_name):
        self.fig.savefig(plot_name)
