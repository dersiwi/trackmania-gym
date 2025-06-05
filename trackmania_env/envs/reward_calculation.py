from __future__ import annotations
from trackmania_env.envs.position_buffer import PositionBuffer
from tminterface.structs import CheckpointData, SimStateData, CheckpointTime
import numpy as np

class RewradCalculator:
    """Responsible for reward calculations for environment"""

    @staticmethod
    def get_instance(reward_calculator : str, position_buffer : PositionBuffer) -> RewradCalculator:

        if reward_calculator == "basic":
            return BasicRewardCalculation(position_buffer)

        else:
            raise NameError(f"Rewardcalculator '{reward_calculator}' not known.")

    def __init__(self, position_buffer : PositionBuffer):
        self.pos_buffer = position_buffer # do not reset or add anything to this position buffer, read-only! (no reset, no add...)

    def calculate_reward(self, observations : dict[str, any], race_finished : bool, other_terminations : bool) -> float:
        """Calculates the rewrad given observations for current environment-step"""
        raise NotImplementedError("Do Not use this class directly. Use RewardCalculator.get_instance()")

    def reset(self) -> None:
        """resets rewrad calculator"""
        pass


class BasicRewardCalculation(RewradCalculator):

    def __init__(self, position_buffer):
        super().__init__(position_buffer)

    def calculate_reward(self, observations, race_finished, other_terminations):
        #ssD : SimStateData = observations["SimStateData"]
        velocity = observations["velocity"]
        reward = np.linalg.norm(velocity[0:1]) + self.pos_buffer.distance_moved()

        if race_finished:
            reward += 10000
        if other_terminations:
            reward -= 1000
        return reward