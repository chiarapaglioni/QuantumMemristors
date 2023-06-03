import matplotlib.pyplot as plt


class IV_plot:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 3)
        self.ax.axhline(0, color='black', linewidth=0.5)
        self.ax.axvline(0, color='black', linewidth=0.5)
        self.xs = []
        self.ys = []
        self.dots, = self.ax.plot([], [], 'o', markersize=3)

    def update(self, v, i):
        # v = v * 100
        # i = -i * 100000
        self.xs.append(v)
        self.ys.append(-i)
        self.xs = self.xs[-2:]
        self.ys = self.ys[-2:]
        # Uncomment the next line if we want to see only one dot at each update
        # self.ax.cla()
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 3)
        self.ax.axhline(0, color='black', linewidth=0.5)
        self.ax.axvline(0, color='black', linewidth=0.5)
        self.ax.plot(self.xs, self.ys, 'o', markersize=3)
        plt.pause(0.001)
