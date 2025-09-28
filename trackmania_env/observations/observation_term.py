from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Space

from tminterface.structs import SimStateData

class ObservationTerm(ABC):
    """
    Base class for observation terms.
    """
    # TODO do we really need the convert torch ?
    def __init__(self, name: str,normalize: bool):
        self.name = name
        self.normalize = normalize
        self.env: Optional[gym.Env] = None
        self.observation_space: Optional[Space] = None
        self.info : Dict[str,any] = {}

    def set_env(self, env: gym.Env):
        self.env = env

    @abstractmethod
    def _get_obs(self, game_states: Dict[str, Union[np.ndarray, SimStateData]], **kwargs) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    # thaught about making this an abstract method but then i had do implement it in every sub class and most of them do nothing on reset
    def reset(self): pass

    def get_observation(
        self, game_states: Dict[str, Union[np.ndarray, SimStateData]], **kwargs) -> tuple[str, np.ndarray]:
        obs = self._get_obs(game_states, **kwargs)
        if self.normalize:
            obs = self._normalize(obs)
        return self.name, obs

    def get_observation_space(self) -> Optional[Space]:
        return self.observation_space

class Obs_Float_Stacker(ObservationTerm):
    """
    Combines multiple ObservationTerms into a single stacked float vector.
    """
    # TODO think of turning this into an obsmanager maybe
    def __init__(self, observation_terms: List[ObservationTerm], name: str = "floats",normalize: bool = False):
        super().__init__(name=name,normalize=normalize)
        self.observation_terms = observation_terms
        self.observation_space = None 

        for term in self.observation_terms:
            term.normalize = normalize

    def set_env(self, env: gym.Env):
        self.env = env
        for term in self.observation_terms:
            term.set_env(env)

    def _get_obs(self, game_states: Dict[str, Union[np.ndarray, SimStateData]], **kwargs) -> np.ndarray:
        obs_List = []
        for term in self.observation_terms:
            _, obs = term.get_observation(game_states)
            obs_List.append(obs.ravel())  # flatten in case of multidimensional obs
        return np.concatenate(obs_List, axis=-1,dtype=np.float32)

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        # This is a bit confusing but the normalisation already happens in _get_obs
        return obs  

    def get_observation_space(self) -> Space:
        if self.observation_space is None:
            sizes = []
            for term in self.observation_terms:
                space = term.get_observation_space()
                assert isinstance(space, spaces.Box), "Only Box spaces are supported for stacking"
                sizes.append(np.prod(space.shape))
            total_size = int(sum(sizes))
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_size,), dtype=np.float32)
        return self.observation_space