
from tminterface.structs import CheckpointData, SimStateData, CheckpointTime

from trackmania_env.rewards.reward_calculation import RewradCalculator
from trackmania_env.rewards.reward_terms.basic_terms import DriveForwardReward
from game_interaction.ipc_fields import IPCFields
import numpy as np

class BasicRewardCalculation(RewradCalculator):

    def __init__(self, normalize : bool = False):
        super().__init__(normalize)

    def get_sum_of_weighted_rewards(self, observations, processed_obs, race_finished, other_terminations):
        ssD : SimStateData = observations[IPCFields.SIMSTATE]
        velocity = ssD.velocity
        reward = np.linalg.norm(velocity[0:1]) + self.pos_buffer.distance_moved()

        if race_finished:
            reward += 10000
        return reward, {}
    

class ForwardReward(RewradCalculator):

    def __init__(self, drive_forward_reward,
                normalize : bool = False,
                **kwargs ):
        """
        Initializes the reward manager with explicit reward weights and parameters
        for modeling centerline distance reward using a Gaussian function.

        Args:
            drive_forward_reward (float):   Weight for forward-acceleration acitons
    """
        
        super().__init__(normalize)

        self.terms = [DriveForwardReward(drive_forward_reward)]
        if len(kwargs) > 0:
            print(f"Got additional kwargsuments that are not used; ignoring them : {kwargs.keys()}. Maybe they're used by a class that inherits.")

    def reset(self):
        self.current_refline_idx = 0
        return super().reset()