import multiprocessing as mp
from queue import Empty
from trackmania_env.plotting.environment_plots import EnvPlotter
from trackmania_env.envs.testcases_single_agent import TestEnvironmentCallback
from abc import ABC, abstractmethod

class PlotterProcess(mp.Process):
    def __init__(self, data_queue:mp.Queue, plotter:EnvPlotter):
        """
        Parameters:
            data_queue (mp.Queue): Queue receiving data to plot.
            plotter (EnvPlotter): An instance of a concrete EnvPlotter subclass.
        """
        super().__init__()
        self.queue:mp.Queue = data_queue
        self.plotter:EnvPlotter = plotter

    def run(self):
        """
        Run the plotting loop in a separate process.
        """
       # self.plotter.setup_plot()

        while True:
            try:
                data = self.queue.get(timeout=1)

                if data is None:
                    print("[PlotterProcess] Shutdown signal received.")
                    break  

                self.plotter.plot(data)

            except Empty: continue

class NonBlockingPlot(TestEnvironmentCallback):
    def __init__(self, plotter):
        super().__init__()
        self.queue = mp.Queue()
        self.plot_process = PlotterProcess(data_queue=self.queue, plotter=plotter)
        self.plot_process.start()

    def __del__(self):
        try:
            self.queue.put(None)  # Signal to shutdown
            self.plot_process.join(timeout=1)
        except Exception:
            pass

class EnvPlotter(ABC):
    """
    Abstract base class for environment plotters.
    Subclasses must implement setup_plot() and plot() methods.
    """

    @abstractmethod
    def setup_plot(self):
        pass

    @abstractmethod
    def plot(self,data):
        pass
