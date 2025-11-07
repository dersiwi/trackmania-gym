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

# TODO convert the observation in torch here 
class ObservationManager(ABC):
    def __init__(self,observation_terms: Union[ObservationTerm,List[ObservationTerm]], convert_torch: bool = True, normalize: bool = False, debug:bool = False):
        """
        Base manager class.
        """

        self.observation_terms = observation_terms
        self.convert_torch = convert_torch
        self.normalize = normalize
        self.debug = debug

        # Environment-related attributes
        self.env: Optional[gym.Env] = None
        self.refline_manager: Optional[ReferenceLineManager] = None
        self.pos_buffer = None  # TODO: purpose of this buffer exactly here? Do some Observation terms need that ?

        # Observation state
        self.obs_space: Optional[Space] = None  # Gym-style observation space
        self.observation = None                 # Final structured observation
        self.info: Dict[str, Any] = {}          # Debugging info

        self._validated = False # One-time validation flag

        self.set_normalize()

    def get_observation_space(self) -> Space:
        """
        Returns the observation space, typically a gym.spaces.Dict or Box.
        """
        return self.obs_space
    
    def set_env(self, env: gym.Env):
        """
        Sets the environment and propagates it to observation terms.
        """
        self.env = env
        self.set_obs_term_env()

    def get_observation(self, obs: Dict[str, Union[np.ndarray, SimStateData]]) -> Tuple[Dict[str, np.ndarray],Dict[str,Any]]:
        """
        Processes raw input like {"image": np.ndarray , "sim_state": SimStateData(...)} into a structured observation
        and debug info using `_get_observation`. Checks format with `check_return_obs`.
        """
        obs, info = self._get_observation(obs)
        self.check_return_obs(obs)
        return obs,info
    
    def check_return_obs(self, obs: Union[np.ndarray, Dict[str, np.ndarray]]):
        """
        Sanity-check that the returned observation matches the observation space format.
        """
        if self.debug or not self._validated:
            self._check_return_obs(obs)
            self._validated = True
    
    # NOTE in the future this typing for the params could be off
    @abstractmethod
    def _check_return_obs(self, obs: Union[np.ndarray, Dict[str, np.ndarray]]):
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
    def _get_observation(self,obs: Dict[str, Union[np.ndarray, SimStateData]]) -> Tuple[Dict[str, np.ndarray],Dict[str,Any]]:
        """
        Internal method that processes raw observations into structured data.
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
        Reset Any internal state (e.g., buffers or stateful observation terms).
        """
        raise NotImplementedError

class BoxObservationManager(ObservationManager):
    """
    Observation Manager who only can handle box-based observation spaces.
    """
    def __init__(self, observation_term: ObservationTerm , convert_torch: bool = True, normalize: bool = False, debug: bool = False):
        assert isinstance(observation_term.observation_space,spaces.Box)
        super().__init__(observation_term, convert_torch, normalize, debug)
        self.obs_space: spaces.Box = self.observation_terms.observation_space

    def _check_return_obs(self, obs: Union[np.ndarray, Dict[str, np.ndarray]]):
        assert isinstance(obs, np.ndarray), f"Expected np.ndarray, got {type(obs)} instead."
        assert self.obs_space.shape == obs.shape, (
        f"Observation shape mismatch: expected {self.obs_space.shape}, got {obs.shape}."
        )

    def set_obs_term_env(self):
        assert self.env is not None, "Environment must be set before assigning to terms."
        self.observation_terms.set_env(self.env)

    def _get_observation(self, obs: Dict[str, Union[np.ndarray, SimStateData]]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        _, computed_obs = self.observation_terms.get_observation(obs)
        self.info.update(self.observation_terms.info)
        return computed_obs,self.info

    def set_normalize(self):
        self.observation_terms.normalize = self.normalize 

    def reset(self):
        self.observation_terms.reset()

class CompositeObservationManager(ObservationManager, ABC):
    """
    Abstract base class for managers that produce composite observations
    from multiple ObservationTerms.
    """
    def __init__(self, observation_terms: List[ObservationTerm], convert_torch = True, normalize = False):
        super().__init__(observation_terms, convert_torch, normalize)
    
    def set_obs_term_env(self):
        assert self.env is not None, "Environment must be set before assigning to terms."
        for term in self.observation_terms:
            term.set_env(self.env)

    def set_normalize(self):
        for term in self.observation_terms:
            term.normalize = self.normalize

    def reset(self):
        for term in self.observation_terms:
            term.reset()

class DictObservationManager(CompositeObservationManager):
    """
    Observation manager for Dictionary-based observation spaces.
    """
    def __init__(self,observation_terms: List[ObservationTerm],convert_torch: bool = True,normalize: bool = False):
        self.check_overriding_obs(observation_terms)
        super().__init__(observation_terms,convert_torch, normalize)
        self.obs_space : spaces.Dict = spaces.Dict({term.name: term.get_observation_space() for term in self.observation_terms})
        self.observation: Dict[str, Optional[np.ndarray]] = {key: None for key in self.obs_space.spaces}
   
    def _get_observation(self, obs: Dict[str, Union[np.ndarray, SimStateData]]) -> Tuple[Dict[str, np.ndarray],Dict[str,Any]]:
        assert self.obs_space is not None, "Observation space not initialized."
        for term in self.observation_terms:
            name, computed_obs = term.get_observation(obs)
            assert name in self.obs_space.spaces, f"Observation '{name}' not in obs_space."
            self.info.update(term.info) 
            self.observation[name] = computed_obs
        return self.observation,self.info


    def _check_return_obs(self, obs: Dict[str, np.ndarray]) -> bool:
        """
        Checks that the returned observation matches the expected observation space.

        Each key in `obs` must exist in `self.obs_space`, and each corresponding value
        must have the same shape and dtype as defined in the space.

        """
        for key, value in obs.items():
            assert key in self.obs_space.spaces, f"Unexpected key in observation: '{key}'"

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
