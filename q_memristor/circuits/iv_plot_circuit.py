import matplotlib.pyplot as plt


class IVplot:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.ax.set_xlim(-0.05, 0.05)
        self.ax.set_ylim(-1, 1)
        self.ax.axhline(0, color='black', linewidth=0.5)
        self.ax.axvline(0, color='black', linewidth=0.5)
        self.xs = []
        self.ys = []
        self.dots, = self.ax.plot([], [], 'o', markersize=3)
        self.ax.set_xlabel('Voltage <V>')
        self.ax.set_ylabel('Current <I>')
        self.ax.set_title('IV Plot')

    def update(self, v, i):
        self.xs.append(v)
        self.ys.append(-i)
        # self.xs = self.xs[-1:]
        # self.ys = self.ys[-1:]
        # Uncomment the next line if we want to see only one dot at each update
        # self.ax.cla()
        self.ax.set_xlim(-0.05, 0.05)
        self.ax.set_ylim(-1, 1)
        self.ax.axhline(0, color='black', linewidth=0.5)
        self.ax.axvline(0, color='black', linewidth=0.5)
        self.ax.plot(self.xs, self.ys, 'o', markersize=3)
        plt.pause(0.001)

    def save_plot(self):
        self.fig.savefig('iv_plot.png')
