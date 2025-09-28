from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
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
    # TODO do we really need the convert torch ?
    def __init__(self, name: str,normalize: bool):
        self.name = name
        self.normalize = normalize
        self.env: Optional[gym.Env] = None
        self.observation_space: Optional[Space] = None
        self.info : dict[str,any] = {}

    def set_env(self, env: gym.Env):
        self.env = env

    @abstractmethod
    def _get_obs(self, game_states: dict[str, Union[np.ndarray, SimStateData]], **kwargs) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    # thaught about making this an abstract method but then i had do implement it in every sub class and most of them do nothing on reset
    def reset(self): pass

    def get_observation(
        self, game_states: dict[str, Union[np.ndarray, SimStateData]], **kwargs) -> tuple[str, np.ndarray]:
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
    def __init__(self, observation_terms: list[ObservationTerm], name: str = "floats",normalize: bool = False):
        super().__init__(name=name,normalize=normalize)
        self.observation_terms = observation_terms
        self.observation_space = None 

        for term in self.observation_terms:
            term.normalize = normalize

    def set_env(self, env: gym.Env):
        self.env = env
        for term in self.observation_terms:
            term.set_env(env)

    def _get_obs(self, game_states: dict[str, Union[np.ndarray, SimStateData]], **kwargs) -> np.ndarray:
        obs_list = []
        for term in self.observation_terms:
            _, obs = term.get_observation(game_states)
            obs_list.append(obs.ravel())  # flatten in case of multidimensional obs
        return np.concatenate(obs_list, axis=-1)

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

# TODO convert the observation in torch here 
class ObservationManager(ABC):
    def __init__(self, convert_torch: bool = True, normalize: bool = False):
        """
        Base manager class for handling multiple ObservationTerm instances.
        """
        self.env: Optional[gym.Env] = None
        self.pos_buffer = None  # Read-only TODO for what do we need this here ?
        self.refline_manager: Optional[ReferenceLineManager] = None

        self.obs_space: Optional[Space] = None # the observations space in gymnasium definition
        self.observation_terms: list[ObservationTerm] = [] 
        self.observation = None # this is what get_observation returns
        self.info: dict[str,any] = {} # for debugging purposes

        self.normalize = normalize
        self.convert_torch = convert_torch

    def set_env(self, env: gym.Env):
        self.env = env
        self.set_obs_term_env()

    @abstractmethod
    def set_obs_term_env(self):
        raise NotImplementedError

    @abstractmethod
    def get_observation(self, obs: dict[str, Union[np.ndarray, SimStateData]]) -> Tuple[dict[str, np.ndarray],dict[str,any]]:
        """
        the function parameter obs is here raw_observation meaning 
        raw_obs = {
            "ssD" : ...
            "image" : ...
        }
        this function returns a tuple (obs,info) where obs are the observations and 
        info are additional infos which could be used for debugging

        """
        raise NotImplementedError

    @abstractmethod
    def get_observation_space(self) -> Space:
        raise NotImplementedError

    @abstractmethod
    def set_normalize(self):
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError


class DictObservationManager(ObservationManager):
    """
    Observation manager for dictionary-based observation spaces.
    """

    def __init__(self,observation_terms: list[ObservationTerm],convert_torch: bool = True,normalize: bool = False):
        super().__init__(convert_torch, normalize)
        self.observation_terms = observation_terms
        self.observation: dict[str,np.ndarray] = {} 

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

    def get_observation(self, obs: dict[str, Union[np.ndarray, SimStateData]]) -> Tuple[dict[str, np.ndarray],dict[str,any]]:
        assert self.obs_space is not None, "Observation space not initialized."
        for term in self.observation_terms:
            name, computed_obs = term.get_observation(obs)
            assert name in self.obs_space.spaces, f"Observation '{name}' not in obs_space."
            self.info.update(term.info) 
            self.observation[name] = computed_obs
        return self.observation,self.info

    def set_normalize(self):
        for term in self.observation_terms:
            term.normalize = self.normalize

    def reset(self):
        for term in self.observation_terms:
            term.reset()