import matplotlib.pyplot as plt


class Tplot:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.ax.axhline(0, color='black', linewidth=0.5)
        self.xs = []
        self.voltages = []
        self.currents = []
        self.preferred_w = 1000
        self.preferred_h = 500
        # self.voltage_scale_factor = 100
        # self.current_scale_factor = 100000
        self.labels_displayed = False
        self.ax.set_xlabel('Time (t)')
        self.ax.set_ylabel('Current <I> & Voltage <V>')
        self.ax.set_title('Time Plot')

    def update(self, t, v, i):
        self.xs.append(t)
        self.voltages.append(v)
        self.currents.append(i)
        # Uncomment the next line if we want to see only one dot at each update
        # self.ax.cla()
        self.ax.axhline(0, color='black', linewidth=0.5)
        self.ax.plot(self.xs, self.voltages, 'bo-', markersize=3, label='Voltage')
        self.ax.plot(self.xs, self.currents, 'ro-', markersize=3, label='Current')

        self.ax.set_xlim(0, t+1)
        self.ax.set_ylim(-0.4, 0.3)
        # self.ax.legend()

        if not self.labels_displayed:
            self.ax.legend()
            self.labels_displayed = True

        plt.pause(0.001)
