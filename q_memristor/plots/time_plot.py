import matplotlib.pyplot as plt


class t_plot:
    def __init__(self, time_steps, ):
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.ax.axhline(0, color='black', linewidth=0.5)
        self.ts = time_steps
        self.xs = []
        self.voltages = []
        self.currents = []
        self.preferred_w = 1000
        self.preferred_h = 500
        # self.voltage_scale_factor = 100
        # self.current_scale_factor = 100000

    def update(self, t, v, i):
        # TODO: implemented voltage and current normalisation
        # v = v * self.voltage_scale_factor
        # i = i * self.current_scale_factor

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

        plt.pause(0.001)
