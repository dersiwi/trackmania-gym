from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union
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
        return np.concatenate(obs_List, axis=-1)

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
    def __init__(self,observation_terms: List[ObservationTerm], convert_torch: bool = True, normalize: bool = False):
        """
        Base manager class for handling multiple ObservationTerm instances.
        """
        self.check_overriding_obs(observation_terms)

        self.observation_terms: List[ObservationTerm] = observation_terms
        self.convert_torch = convert_torch
        self.normalize = normalize

        # Environment-related attributes
        self.env: Optional[gym.Env] = None
        self.refline_manager: Optional[ReferenceLineManager] = None
        self.pos_buffer = None  # TODO: purpose of this buffer exactly here? Do some Observation terms need that

        # Observation state
        self.obs_space: Optional[Space] = None  # Gym-style observation space
        self.observation = None                 # Final structured observation
        self.info: Dict[str, Any] = {}          # Debugging info

    def set_env(self, env: gym.Env):
        """
        Sets the environment and propagates it to observation terms.
        """
        self.env = env
        self.set_obs_term_env()

    def check_overriding_obs(self, obs_terms: List[ObservationTerm]):
        """
        Check if there are observation terms with duplicate names.
        This is important because duplicates could override each other 
        later in processing, leading to unexpected behavior.
        """
        obs_term_names = [t.name for t in obs_terms]
        name_counts = Counter(obs_term_names)
        duplicates = [name for name, count in name_counts.items() if count > 1]
    
        if duplicates:
            raise ValueError(
                f"Duplicate observation terms found: {duplicates}. "
                "Each observation term must have a unique name to avoid overriding."
            )
    
    def get_observation(self, obs: Dict[str, Union[np.ndarray, SimStateData]]) -> Tuple[Dict[str, np.ndarray],Dict[str,any]]:
        """
        Processes raw input like {"image": np.ndarray , "sim_state": SimStateData(...)} into a structured observation
        and debug info using `_get_observation`. Checks format with `check_return_obs`.
        """
        obs, info = self._get_observation(obs)
        self.check_return_obs(obs)
        return obs,info
    
    @abstractmethod
    def check_return_obs(self, obs: Union[np.ndarray, Dict[str, np.ndarray]]):
        """
        Sanity-check that the returned observation matches the observation space format.
        """
        raise NotImplementedError

    @abstractmethod
    def set_obs_term_env(self):
        """
        Propagates the environment to each observation term.
        """
        raise NotImplementedError
    
    @abstractmethod
    def _get_observation(self,obs: Dict[str, Union[np.ndarray, SimStateData]]) -> Tuple[Dict[str, np.ndarray],Dict[str,any]]:
        """
        Internal method that processes raw observations into structured data.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_observation_space(self) -> Space:
        """
        Returns the observation space, typically a gym.spaces.Dict or Box.
        """
        raise NotImplementedError

    @abstractmethod
    def set_normalize(self):
        """
        Optionally enable or configure observation normalization.
        """
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        """
        Reset any internal state (e.g., buffers or stateful observation terms).
        """
        raise NotImplementedError
    



class DictObservationManager(ObservationManager):
    """
    Observation manager for Dictionary-based observation spaces.
    """

    def __init__(self,observation_terms: List[ObservationTerm],convert_torch: bool = True,normalize: bool = False):
        super().__init__(convert_torch, normalize)
        self.observation: Dict[str,np.ndarray] = {} 

    def set_obs_term_env(self):
        assert self.env is not None, "Environment must be set before assigning to terms."
        for term in self.observation_terms:
            term.set_env(self.env)

    def get_observation_space(self) -> Space:
        if self.obs_space is None:
            space_Dict = {
                term.name: term.get_observation_space() for term in self.observation_terms
            }
            self.obs_space = spaces.Dict(space_Dict)
        return self.obs_space

    def _get_observation(self, obs: Dict[str, Union[np.ndarray, SimStateData]]) -> Tuple[Dict[str, np.ndarray],Dict[str,any]]:
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

    def check_return_obs(self, obs: Dict[str, np.ndarray]) -> bool:
        """
        Checks that the returned observation matches the expected observation space.

        Each key in `obs` must exist in `self.obs_space`, and each corresponding value
        must have the same shape and dtype as defined in the space.

        """
        for key, value in obs.items():
            assert key in self.obs_space, f"Unexpected key in observation: '{key}'"

            expected_space = self.obs_space[key]

            # Check shape
            assert value.shape == expected_space.shape, (
                f"Shape mismatch for key '{key}': "
                f"expected {expected_space.shape}, got {value.shape}"
            )

            # Check dtype
            assert value.dtype == expected_space.dtype, (
                f"Dtype mismatch for key '{key}': "
                f"expected {expected_space.dtype}, got {value.dtype}"
            )