import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib import rc
rc('text', usetex=True)


class Arm:

    def __init__(self, ax, color="black", zorder=0, alpha=1):

        self.ax = ax
        self.arm, = self.ax.plot([0, 0], [0, 1], color=color,
                                 lw=3, zorder=zorder, alpha=alpha)
        self.hand = self.ax.scatter(0, 1, color=color, s=200,
                                    zorder=zorder, alpha=alpha)

        for a in np.linspace(0, 2*np.pi, 5):
            self.ax.plot([0, np.cos(a)], [0, np.sin(a)], color="black", lw=0.5)

    def update(self, arm_angle, hand_pos, secs=0.001):

        self.arm.set_data([0, np.cos(arm_angle)], [0, np.sin(arm_angle)])
        self.hand.set_offsets(hand_pos)


class Plotter:

    def __init__(self, time_window=2000, title=None, ioff=True):

        if ioff:
            plt.ioff()
        else:
            plt.ion()
            
        gs = gridspec.GridSpec(8, 4)

        self.figure = plt.figure(figsize=(4, 8))
        self.ax = self.figure.add_subplot(gs[:6, :], aspect="equal")
        self.ax.set_xlim([-1.5, 1.5])
        self.ax.set_ylim([-1.5, 2])

        if title is not None:
            self.ax.set_title(title)

        self.real_arm = Arm(self.ax, zorder=0, alpha=1)
        self.sensed_arm = Arm(self.ax, "#0000aa", zorder=10, alpha=0.4)
        self.generated_arm = Arm(self.ax, "#aa0000", zorder=10, alpha=0.4)
        self.target_arm = Arm(self.ax, "#00aa00", zorder=10, alpha=1)

        self.ax.legend([self.real_arm.arm,
                        self.sensed_arm.arm,
                        self.generated_arm.arm,
                        self.target_arm.arm],
                       ["real",
                        "perceived",
                        "generated from model",
                        "target"])

        self.time_window = time_window
        self.ax_logs = self.figure.add_subplot(gs[6:, :], aspect="auto")
        self.ax_logs.set_xlim([0, self.time_window])
        self.ax_logs.set_ylim([-0.5*np.pi, 0.5*np.pi])
        self.ax_logs.set_xticks([])
        self.ax_logs.set_yticks([-0.5*np.pi, 0, 0.5*np.pi])
        self.ax_logs.set_yticklabels(
            [r"$-\frac{\pi}{2}$", "$0$", r"$\frac{\pi}{2}$"])

        self.real_mu, = self.ax_logs.plot(0, 0, color="black", lw=1, zorder=20)
        self.model_mu, = self.ax_logs.plot(0, 0, color="red", lw=2)

        self.ax_logs.legend([self.real_mu, self.model_mu],
                            ["real $\mu$", "model $\mu$"])
        self.store_real_mu = []
        self.store_model_mu = []

        plt.tight_layout()

    def append_mu(self, real_mu, model_mu):

        self.store_real_mu.append(real_mu)
        self.store_model_mu.append(model_mu)
        self.store_real_mu = self.store_real_mu[-self.time_window:]
        self.store_model_mu = self.store_model_mu[-self.time_window:]

    def update(self):
        try:
            self.real_mu.set_data(np.arange(len(self.store_real_mu)),
                                  self.store_real_mu)
            self.model_mu.set_data(np.arange(len(self.store_model_mu)),
                                   self.store_model_mu)
            self.figure.canvas.draw()
        except ValueError:
            pass
        self.figure.canvas.draw()
        plt.pause(0.01)
