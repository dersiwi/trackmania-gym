from plotting.core import EnvPlotter
from typing import Type, Dict
from plotting.backends import matplot, pyqtgraph, tmnf_cv2

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
