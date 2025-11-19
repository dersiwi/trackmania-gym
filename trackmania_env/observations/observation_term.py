from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Space

from tminterface.structs import SimStateData



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
        
        from trackmania_env.envs.enivonrments import TMNF_Single_Agent_Env
        self.env: TMNF_Single_Agent_Env = None
        self.observation_space: Optional[Space] = None
        self.info : Dict[str,Any] = {}

    def set_env(self, env):
        """
        Assign the environment to the observation term.
        """
        from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
        self.env : TMNF_Single_Agent_Env = env

    def set_observation_space_as_box(self, low : float, high : float, shape : tuple[int]):
        self.observation_space = spaces.Box(low = low, high = high, shape=shape )

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
    
    def get_observation(self, game_states: Dict[str, Union[np.ndarray, SimStateData]], **kwargs) -> tuple[np.ndarray, dict]:
        """
        Public method to compute the observation, with optional normalization.

        Args:
            game_states (Dict[str, Union[np.ndarray, SimStateData]]): 
                The input state dict {image, sim state}.

        Returns:
            Tuple[str, np.ndarray]: A tuple (observation, info).
        """
        obs, info = self._get_obs(game_states, **kwargs)
        if self.normalize:
            obs = self._normalize(obs)
        return obs, info

    @abstractmethod
    def _get_obs(self, game_states: Dict[str, Union[np.ndarray, SimStateData]], **kwargs) -> tuple[np.ndarray, dict[str, str]]:
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
    
    @abstractmethod
    def flatten(self, processed_obs: np.ndarray) -> np.ndarray:
        """
        Flattens observation into shape (dim,) where dim is the flattened dimension of the term
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_flatten_dim(self) -> int:
        """
        Returns dim, which is the dimension of the flattened observation-term (@see flatten(self))
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_native_shape(self) -> tuple:
        """
        Returns the native shape, such that this term, which has been previously flattened, could be turned back into its
        native shape (for images e.g. this is C, H, W).
        """
        raise NotImplementedError()
    

class VectorlikeTerm(ObservationTerm, ABC):
    """This is an observation term that is naturally in vector-from. A position e.g. lives naturally in a singular-column space. However an image naturally lives in 
    image space. """
    def __init__(self, name, normalize, dimension, low : float = 0.0, high : float = 1.0):
        super().__init__(name, normalize)
        self.dimension = dimension
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(dimension,),
            dtype=np.float32
        )

    def get_flatten_dim(self):
        return self.dimension
    
    def flatten(self, processed_obs):
        return processed_obs
    
    def get_native_shape(self):
        return (self.dimension, )


class GroupedObservationTerm(ObservationTerm):
    """
    Groups multiple ObservationTerms and outputs a single flat array under one name.
    Useful for stacking features into a single Box space.
    """

    def __init__(self, observation_terms: List[VectorlikeTerm], name: str, normalize: bool = True):
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
            obs, info = term.get_observation(game_states)
            obs_list.append(obs.ravel())
            self.info.update(info)

        return np.concatenate(obs_list, axis=-1).astype(np.float32), self.info

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        return obs  # Already normalized at term level
    
    def flatten(self, processed_obs):
        return processed_obs
    
    def get_flatten_dim(self):
        return self.observation_space.shape[0]
    
    def get_native_shape(self):
        return (self.get_flatten_dim(), )

