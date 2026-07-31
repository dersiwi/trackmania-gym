from __future__ import annotations
import multiprocessing as mp
from queue import Empty
from plotting.test_environment_callbacks.core import TestEnvironmentCallback
from typing import Type, Dict
from plotting.backends import matplot, pyqtgraph, tmnf_cv2
from plotting.plotter import EnvPlotter


factories: Dict[str,dict[str,Type[EnvPlotter]]] = {
        "image" : {
            "matplotlib" : matplot.Plot_Obs_Images,
            "pyqtgraph" : pyqtgraph.Plot_Obs_Images,
            "cv2": tmnf_cv2.Plot_Obs_Images,
            },
        "lines" : {
            "matplotlib": matplot.LinePlotter,
            },
        "lateral_distance": {
            "matplotlib" : matplot.Plot_Lateral_Distance,
            },
        "lateral_distance2": {
            "matplotlib" : matplot.Plot_Lateral_Distance2,
            },
        "lateral_distance3": {
            "matplotlib" : matplot.Plot_Lateral_Distance_MapAndGraph,
            },
        "ref_line": {
            "matplotlib": matplot.Plot_RefLine,
            },
        "rotation": {
            "matplotlib": matplot.Rotation_Plotter
            },
        }

class PlottingFactory:
    """
    Create an EnvPlotter instance for a given factory name and backend.
    """

    def __init__(self, factory_name: str, backend: str = "matplotlib") -> None:
        self.factory_name = factory_name
        self.backend = backend

        if factory_name not in factories:
            raise ValueError(f"Unknown factory name: {factory_name!r}")
        if self.backend not in factories[factory_name]:
            available = ", ".join(factories[factory_name].keys())
            raise ValueError(
                    f"Backend {self.backend!r} not registered for factory {factory_name!r}. "
                    f"Available: {available}"
                    )

    def create(self, **kwargs) -> EnvPlotter:
        """
        Instantiate the concrete EnvPlotter and pass any extra kwargs).

        Parameters
        ----------
        env : Any
            The TrackMania environment instance that the plotter will visualise.
        **kwargs
            Extra keyword arguments forwarded to the plotter's ``__init__``.
        """
        cls = factories[self.factory_name][self.backend]
        return cls(**kwargs)




class PlotterProcess(mp.Process):
    def __init__(self, data_queue:mp.Queue, factory_name : str, backend : str, create_args : dict[str, any]):
        """
        Args:
            data_queue (mp.Queue)   : Queue receiving data to plot.
            factory_name (str)      : Name of the factory creating the plotter
            backend (str)           : Backend to use
            create_args (dict)      : Arguments passed to the .create() method of the factory. Plotter-Dependent.
        """
        super().__init__()
        self.queue:mp.Queue = data_queue
        self.factory_name = factory_name
        self.backend = backend
        self.create_args = create_args
        self.plotter:EnvPlotter = None # is only created in the run-method

    def run(self):
        """
        Run the plotting loop in a separate process.
        """
       # self.plotter.setup_plot()
        self.plotter = PlottingFactory(factory_name = self.factory_name, backend = self.backend).create(**self.create_args)
        while True:
            try:
                data = self.queue.get(timeout=1)

                if data is None:
                    print("[PlotterProcess] Shutdown signal received.")
                    break  

                self.plotter.plot(data)

            except Empty: continue

class NonBlockingPlot(TestEnvironmentCallback):
    def __init__(self, factory_name : str, backend : str, create_args):
        super().__init__()
        self.queue = mp.Queue()
        self.plot_process = PlotterProcess(data_queue=self.queue, factory_name=factory_name, backend=backend, create_args = create_args)
        self.plot_process.start()

    def __del__(self):
        try:
            self.queue.put(None)  # Signal to shutdown
            self.plot_process.join(timeout=1)
        except Exception:
            pass
