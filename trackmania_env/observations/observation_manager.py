from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Space

from tminterface.structs import SimStateData
from trackmania_env.utils.reference_line_manager import ReferenceLineManager

class ObservationTerm(ABC):
    """
    Base class for observation terms.
    """

    def __init__(self, name: str, convert_torch: bool, normalize: bool):
        self.name = name
        self.convert_torch = convert_torch
        self.normalize = normalize
        self.env: Optional[gym.Env] = None
        self.observation_space: Optional[Space] = None

    def set_env(self, env: gym.Env):
        self.env = env

    @abstractmethod
    def _get_obs(self, game_states: dict[str, Union[np.ndarray, "SimStateData"]], **kwargs) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def get_observation(
        self, game_states: dict[str, Union[np.ndarray, "SimStateData"]], **kwargs) -> tuple[str, np.ndarray]:
        obs = self._get_obs(game_states, **kwargs)
        if self.normalize:
            obs = self._normalize(obs)
        return self.name, obs

    def get_observation_space(self) -> Optional[Space]:
        return self.observation_space


class ObservationManager(ABC):
    def __init__(self, convert_torch: bool = True, normalize: bool = False):
        """
        Base manager class for handling multiple ObservationTerm instances.
        """
        self.env: Optional[gym.Env] = None
        self.pos_buffer = None  # Read-only
        self.refline_manager: Optional[ReferenceLineManager] = None

        self.obs_space: Optional[Space] = None
        self.observation_terms: list[ObservationTerm] = []

        self.normalize = normalize
        self.convert_torch = convert_torch

    def set_env(self, env: gym.Env):
        self.env = env
        self.set_obs_term_env()

    @abstractmethod
    def set_obs_term_env(self):
        raise NotImplementedError

    @abstractmethod
    def get_observation(self, obs: dict[str, Union[np.ndarray, SimStateData]]) -> dict[str, np.ndarray]:
        """
        obs are here raw_observation meaning 
        raw_obs = {
            "ssD" : ...
            "image" : ...
        }
        """
        raise NotImplementedError

    @abstractmethod
    def get_observation_space(self) -> Space:
        raise NotImplementedError

    @abstractmethod
    def set_normalize(self):
        raise NotImplementedError

    @abstractmethod
    def set_convert_torch(self):
        raise NotImplementedError


class DictObservationManager(ObservationManager):
    """
    Observation manager for dictionary-based observation spaces.
    """

    def __init__(self,observation_terms: list[ObservationTerm],convert_torch: bool = True,normalize: bool = False):
        super().__init__(convert_torch, normalize)
        self.observation_terms = observation_terms

    def set_obs_term_env(self):
        assert self.env is not None, "Environment must be set before assigning to terms."
        for term in self.observation_terms:
            term.set_env(self.env)

    def get_observation_space(self) -> Space:
        if self.obs_space is None:
            space_dict = {
                term.name: term.get_observation_space() for term in self.observation_terms
            }
            self.obs_space = spaces.Dict(space_dict)
        return self.obs_space

    def get_observation(self, obs: dict[str, Union[np.ndarray, "SimStateData"]]) -> dict[str, np.ndarray]:
        assert self.obs_space is not None, "Observation space not initialized."
        observations = {}
        for term in self.observation_terms:
            name, computed_obs = term.get_observation(obs)
            assert name in self.obs_space.spaces, f"Observation '{name}' not in obs_space."
            observations[name] = computed_obs
        return observations

    def set_normalize(self):
        for term in self.observation_terms:
            term.normalize = self.normalize

    def set_convert_torch(self):
        for term in self.observation_terms:
            term.convert_torch = self.convert_torch
