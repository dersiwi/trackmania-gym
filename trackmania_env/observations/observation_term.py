from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Space

from tminterface.structs import SimStateData

# TODO should we include the option for setting the dtype ? via passing it as a param through the constructor 
class ObservationTerm(ABC):
    """
    Base class for individual observation terms used in Trackmania environments.
    """
    def __init__(self, name: str, normalize: bool):
        """
        Args:
            name (str): The name of this observation term.
            normalize (bool): Whether to normalize this observation.
        """
        self.name = name
        self.normalize = normalize

        self.env: Optional[gym.Env] = None
        self.observation_space: Optional[Space] = None
        self.info : Dict[str,Any] = {}

    def set_env(self, env: gym.Env):
        """
        Assign the environment to the observation term.
        """
        self.env = env

    def reset(self):
        """
        Optional reset hook for observation terms that maintain internal state.
        Override in subclasses if needed.
        We made it do nothing on purpose since most of the subclasses dont actually do anything during reset
        """
        ...

    def get_observation_space(self) -> Optional[Space]:
        """
        Returns the observation space for this term.
        Should be set in the subclass during __init__.
        """
        return self.observation_space
    
    def get_observation(self, game_states: Dict[str, Union[np.ndarray, SimStateData]], **kwargs) -> tuple[str, np.ndarray]:
        """
        Public method to compute the observation, with optional normalization.

        Args:
            game_states (Dict[str, Union[np.ndarray, SimStateData]]): 
                The input state dict {image, sim state}.

        Returns:
            Tuple[str, np.ndarray]: A tuple (name, observation).
        """
        obs = self._get_obs(game_states, **kwargs)
        if self.normalize:
            obs = self._normalize(obs)
        return self.name, obs

    @abstractmethod
    def _get_obs(self, game_states: Dict[str, Union[np.ndarray, SimStateData]], **kwargs) -> np.ndarray:
        """
        Compute the unnormalized observation.
        Must be implemented by all subclasses.
        """
        raise NotImplementedError()

    @abstractmethod
    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize the observation returned by `_get_obs()` if `self.normalize` is True.
        Must be implemented by all subclasses.
        """
        raise NotImplementedError()
    

class GroupedObservationTerm(ObservationTerm):
    """
    Groups multiple ObservationTerms and outputs a single flat array under one name.
    Useful for stacking features into a single Box space.
    """

    def __init__(self, observation_terms: List[ObservationTerm], name: str, normalize: bool = False):
        super().__init__(name=name, normalize=normalize)
        self.observation_terms = observation_terms

        sizes = []
        for term in observation_terms:
            space = term.get_observation_space()
            assert isinstance(space, spaces.Box), "Only Box spaces are supported for stacking"
            sizes.append(np.prod(space.shape))
        
        total_size = int(sum(sizes))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_size,), dtype=np.float32)

        # Normalize must propagate to children
        for term in self.observation_terms:
            term.normalize = normalize

    def set_env(self, env: gym.Env):
        super().set_env(env)
        for term in self.observation_terms:
            term.set_env(env)

    def reset(self):
        for term in self.observation_terms:
            term.reset()

    def _get_obs(self, game_states: Dict[str, Union[np.ndarray, SimStateData]], **kwargs) -> np.ndarray:
        obs_list = []
        for term in self.observation_terms:
            obs = term.get_observation(game_states)[1].ravel()
            obs_list.append(obs)
            self.info.update(term.info)

        return np.concatenate(obs_list, axis=-1).astype(np.float32)

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        return obs  # Already normalized at term level
