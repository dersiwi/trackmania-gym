class TestEnvironmentCallback():
    """TestEnviornmentCallbacks are used to track, log, do whatever with data obtained by an environment per setp."""

    def __init__(self):
        self.n_step = 0
        """Counts environment-steps aka. how often _call_after_step was called."""

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        """This method is called by TestEnvironment.step_with_manual_input(), after everytime this method executes
        an environment step of the underlying environment."""
        pass

    def _call_after_run(self):
        """This method is called by TestEnvironment.step_with_manual_input(), after the main-loop has been executed via `esc`."""
        pass

    def reset(self):
        """Resets the callback, if the user presses 'r'"""
        pass

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