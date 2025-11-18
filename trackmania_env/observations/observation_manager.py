from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Space

from tminterface.structs import SimStateData
from trackmania_env.utils.reference_line_manager import ReferenceLineManager
from trackmania_env.observations.observation_term import ObservationTerm
from trackmania_env.utils.spacetransform import SpaceTransformer

class ObservationManager(ABC):
    def __init__(self,observation_terms: list[ObservationTerm], convert_torch : bool = True, normalize : bool = False, return_as_dict : bool = True):
        """
        Base manager class.
        """
        
        self.observation_terms = observation_terms
        self.convert_torch = convert_torch
        self.normalize = normalize
        self.return_as_dict = return_as_dict
        
        # Environment-related attributes
        from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
        self.env : TMNF_Single_Agent_Env = None
        self.spacetransformer = SpaceTransformer.init_instance(self.observation_terms)

        # Observation state
        self.obs_space: Optional[Space] = None  # Gym-style observation space

        if self.return_as_dict:
            spacedict = {}
            for term in self.observation_terms:
                spacedict[term] = term.observation_space
            self.obs_space = spaces.Dict(spacedict)
        else:
            self.obs_space = spaces.Box(-np.inf, np.inf, shape=(sum([term.get_flatten_dim() for term in self.observation_terms])), dtype=np.float32)



    def get_observation_space(self) -> Space:
        """
        Returns the observation space, a gym.spaces.Dict or Box.
        """
        return self.obs_space
    
    def set_env(self, env: gym.Env):
        """
        Sets the environment and propagates it to observation terms.
        """
        from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
        self.env : TMNF_Single_Agent_Env = env
        for obsterm in self.observation_terms:
            obsterm.set_env(env)

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
        for obsterm in self.observation_terms:
            observations[obsterm.name], info[obsterm.name] = obsterm.get_observation(obs)
        return observations, info
    
    def reset(self):
        """
        Reset Any internal state (e.g., buffers or stateful observation terms).
        """
        for obsterm in self.observation_terms:
            obsterm.reset()
