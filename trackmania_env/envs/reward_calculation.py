
from trackmania_env.envs.position_buffer import PositionBuffer
from tminterface.structs import CheckpointData, SimStateData, CheckpointTime
import numpy as np

class RewradCalculator:
    """Responsible for reward calculations for environment"""

    def __init__(self, position_buffer : PositionBuffer):
        self.pos_buffer = position_buffer # do not reset or add anything to this position buffer, read-only! (no reset, no add...)

    def calculate_reward(self, observations : dict[str, any]) -> float:
        """Calculates the rewrad given observations for current environment-step"""
        
        ssD : SimStateData = observations["SimStateData"]
        reward = np.linalg.norm(ssD.velocity) + self.pos_buffer.distance_moved()
        return reward

    def reset(self) -> None:
        """resets rewrad calculator"""
        pass