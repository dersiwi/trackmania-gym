from abc import ABC, abstractmethod

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