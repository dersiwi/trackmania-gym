from matplotlib import pyplot as plt
from trackmania_gym.trackmania_env.envs.testenv_single_agent import TestEnvironmentCallback


class Live3dPlotEnvironmentCallback(TestEnvironmentCallback):

    def __init__(self):
        # Set up interactive plot
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Start the plot
        self._setup_plot()
        plt.ion()
        plt.show()

    def _setup_plot(self):
        """Responsible for settingup"""
        pass

    def reset(self):
        """clear the axis"""
        plt.gca().cla()