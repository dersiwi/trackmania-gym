import numpy as np
import os

from gymnasium import spaces
from gymnasium.spaces import Space
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

from tminterface.structs import SimStateData
from trackmania_env.utils.reference_line_manager import ReferenceLineManager
from trackmania_env.observations.observation_term import ObservationTerm
from trackmania_env.observations.observation_terms.img_terms import ImageObservationTerm
from trackmania_env.utils.spacetransform import SpaceTransformer

from trackmania_env.manager import Manager
class ObservationManager(ABC, Manager):
    def __init__(self, observation_terms : list[ObservationTerm], convert_torch : bool = True, normalize : bool = False, return_as_dict : bool = True):
        """
        Base manager class.
        """
        super().__init__()
        self.terms : list[ObservationTerm] = observation_terms
        self.convert_torch = convert_torch
        self.normalize = normalize
        self.return_as_dict = return_as_dict
        
        self.spacetransformer = SpaceTransformer.init_instance(self.terms)

        # Observation state
        self.obs_space: Optional[Space] = None  # Gym-style observation space

        if self.return_as_dict:
            spacedict = {}
            for term in self.terms:
                spacedict[term.name] = term.observation_space
            self.obs_space = spaces.Dict(spacedict)
        else:
            self.obs_space = spaces.Box(-np.inf, np.inf, shape=(sum([term.get_flatten_dim() for term in self.terms])), dtype=np.float32)
        
        # set control-image logging
        controlimgdir, df = os.path.join(os.getcwd(), "logs/control_images"), 1000000
        self.logger.info(f"Setting control-image dumping directory to : '{controlimgdir}' and dumping freq to {df} env steps.")
        for term in self.terms:
            if isinstance(term, ImageObservationTerm):
                term: ImageObservationTerm
                term.set_dump_freq(freq = df, dirpath=controlimgdir)

    def get_observation_space(self) -> Space:
        """
        Returns the observation space, a gym.spaces.Dict or Box.
        """
        return self.obs_space

    def get_observation(self, obs: Dict[str, Union[np.ndarray, SimStateData]]) -> Tuple[Dict[str, np.ndarray],Dict[str,Any]]:
        """
        Processes raw input like {"image": np.ndarray , "sim_state": SimStateData(...)} into a structured observation. NOTE: the raw
        observations are never vectorized.
        """
        obsdict, info = self._get_obs_as_dict(obs)
        if not self.return_as_dict:
            return self.spacetransformer.dict_to_numpy(obsdict), info
        return obsdict, info

    def _get_obs_as_dict(self, obs) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        observations = {}
        info = {}
        
        for obsterm in self.terms:
            try:
                observations[obsterm.name], terminfo = obsterm.get_observation(obs)
            except Exception as e:
                self.logger.error(f"Failure to retrieve observation-term {obsterm.name}, error-traceback : {e} \n\n Supplementing obsterm with zeros.")
                observations[obsterm.name], terminfo = np.zeros(obsterm.get_native_shape(), dtype = np.float32), {}
            info = info | terminfo
        return observations, info