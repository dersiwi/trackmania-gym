
from tminterface.structs import CheckpointData, SimStateData, CheckpointTime

from trackmania_env.rewards.reward_calculation import RewradCalculator
from game_interaction.ipc_fields import IPCFields
import numpy as np

class BasicRewardCalculation(RewradCalculator):

    def __init__(self):
        super().__init__()

    def calculate_reward(self, observations, processed_obs : dict[str, any], race_finished, other_terminations) -> tuple[float, dict[str, any]]:
        ssD : SimStateData = observations[IPCFields.SIMSTATE]
        velocity = ssD.velocity
        reward = np.linalg.norm(velocity[0:1]) + self.pos_buffer.distance_moved()

        if race_finished:
            reward += 10000
        return reward, {}