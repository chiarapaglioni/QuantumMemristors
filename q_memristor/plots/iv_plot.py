import matplotlib.pyplot as plt

"""
    Basic implementation of IV-Plot 

    Author: Chiara Paglioni
"""


class IVplot:
    def __init__(self, v0, i0, bool_dyn):
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.bool_dyn = bool_dyn

        if bool_dyn:
            self.ax.set_xlim(-1.5, 1.5)
            self.ax.set_ylim(-1.5, 1.5)
        else:
            self.ax.set_xlim(-0.3, 0.3)
            self.ax.set_ylim(-0.2, 0.2)

        self.ax.axhline(0, color='black', linewidth=0.5)
        self.ax.axvline(0, color='black', linewidth=0.5)
        self.v0 = v0
        self.i0 = i0
        self.xs = []
        self.ys = []

        self.ax.set_xlabel('Voltage <V>')
        self.ax.set_ylabel('Current <I>')
        self.ax.set_title('IV Plot')

    def update(self, v, i, bool_end):
        # v = v / self.v0
        # i = -i / self.i0
        self.xs.append(v)
        self.ys.append(-i)

        # Uncomment to plot a single dot at each iteration
        # self.xs = self.xs[-1:]
        # self.ys = self.ys[-1:]

        # Uncomment the next line if we want to see only one dot at each update
        # self.ax.cla()

        if self.bool_dyn:
            self.ax.set_xlim(-1.5, 1.5)
            self.ax.set_ylim(-1.5, 1.5)
        else:
            self.ax.set_xlim(-0.3, 0.3)
            self.ax.set_ylim(-0.2, 0.2)

        self.ax.axhline(0, color='black', linewidth=0.5)
        self.ax.axvline(0, color='black', linewidth=0.5)
        self.ax.plot(self.xs, self.ys, 'o', markersize=1)

        # Plot initial point
        self.ax.plot(self.v0, -self.i0, 'ro', markersize=2)

        # Plot final point
        if bool_end:
            self.ax.plot(v, -i, 'bo', markersize=2)
        plt.pause(0.001)

    def save_plot(self):
        if self.bool_dyn:
            self.fig.savefig('figures/iv_plot_dyn.png')
        else:
            self.fig.savefig('figures/iv_plot_stat.png')
