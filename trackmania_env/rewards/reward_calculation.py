from __future__ import annotations
from trackmania_env.utils.position_buffer import PositionBuffer
from tminterface.structs import CheckpointData, SimStateData, CheckpointTime
from game_interaction.ipc_fields import IPCFields
import numpy as np
from numba import jit

class RewradCalculator:
    """Responsible for reward calculations for environment"""

    def __init__(self):
        self.pos_buffer = None # do not reset or add anything to this position buffer, read-only! (no reset, no add...)

    def set_position_buffer(self, position_buffer : PositionBuffer):
        """Set position buffer for this instance"""
        self.pos_buffer = position_buffer

    def calculate_reward(self, observations : dict[str, any], race_finished : bool, other_terminations : bool) -> float:
        """Calculates the rewrad given observations for current environment-step"""
        raise NotImplementedError("Do Not use this class directly. Use RewardCalculator.get_instance()")

    def reset(self) -> None:
        """resets rewrad calculator"""
        pass