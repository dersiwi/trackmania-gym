from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, Tuple, Union, Deque
import numpy as np
from gymnasium.spaces import Box

from trackmania_env.observations.observation_term import ObservationTerm
from game_interaction.ipc_fields import IPCFields
from tminterface.structs import SimStateData
from trackmania_env.utils.actionmap import ACTION_MAP,REVERSE_ACTION_MAP

class HistoryObservationTerm(ObservationTerm, ABC):
    """
    Abstract class for observation terms that maintain a history buffer (queue)
    of previous observations. Subclasses must define the type of observation,
    how to extract it, and how to convert the history into a NumPy array.
    """

    def __init__(self, name: str, normalize: bool, maxlen_history: int):
        super().__init__(name, normalize)

        self.maxlen_history = maxlen_history
        self.init_val, self.dtype = self._initial_value_and_dtype()
        self.hist: Deque[Any] = deque(
            [self.init_val] * maxlen_history, maxlen=maxlen_history
        )

    @abstractmethod
    def _initial_value_and_dtype(self) -> Tuple[Any, np.dtype]:
        """
        Subclass must return:
            - An initial value of the same type as observations it will add
            - The NumPy dtype to use when converting to arrays
        Example:
            >>> return 0.0, np.float32
            >>> return np.zeros(3, dtype=np.float32), np.float32
        """
        raise NotImplementedError

    @abstractmethod
    def extract_obs(self, game_states: Dict[str, Union[np.ndarray, Any]]) -> Any:
        """
        Extract the current observation from the game state.
        Must return the same type as `_initial_value()`.
        """
        raise NotImplementedError

    @abstractmethod
    def _queue_to_array(self) -> np.ndarray:
        """
        Convert the current queue (`self.hist`) to a NumPy array that
        will be returned by the Gym environment.
        This must be implemented to support custom formatting or stacking.
        """
        raise NotImplementedError

    def _get_obs(self, game_states: Dict[str, Union[np.ndarray, Any]], **kwargs) -> np.ndarray:
        new_obs = self.extract_obs(game_states)
        self.hist.append(new_obs)
        return self._queue_to_array(), {}

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        """
        Optionally normalize the observation (if enabled).
        Subclasses can override this if needed.
        """
        return obs

    def reset(self):
        """
        Resets the internal queue to its initial state.
        """
        self.hist = deque([self.init_val] * self.maxlen_history, maxlen=self.maxlen_history)


class DiscreteActionHistoryTerm(HistoryObservationTerm):
    """This observation terms stores only the history of action indicies and not the mappinf to key presses"""
    def __init__(self,maxlen_history, normalize=False , name ="action idx history"):
        super().__init__(name, normalize, maxlen_history)
        self.observation_space = Box(
            low= 0,
            high= len(REVERSE_ACTION_MAP)-1,
            shape= (self.maxlen_history,),
            dtype= self.dtype
        )

    def _initial_value_and_dtype(self):
        dtype = np.float32 #NOTE: could also be just int but most of the observations are floats
        init_val = 0
        return init_val,dtype
    
    def extract_obs(self, game_states):
        action: Tuple[bool, bool, bool, bool] = game_states[IPCFields.ACTION]
        action_ix = REVERSE_ACTION_MAP[action]
        return action_ix
    
    def _queue_to_array(self):
         return np.array(self.hist, dtype=self.dtype)