from trackmania_env.rewards.reward_calculation import RewardTerm, BoundedRewardterm
from trackmania_env.utils import constants
from trackmania_env.utils.lateral_distance_manager import LateralDistanceManager
from game_interaction.ipc_fields import IPCFields
from tminterface.structs import SimStateData, SceneVehicleCar
import numpy as np

class AccumulatedDistanceConstant(RewardTerm):
    NAME = "accumulated_distance"
    def __init__(self, weight, total_reward : float):
        """Iniitialites
        Args:
            weight (float)  : Weight of the term
            total_reward (float)    : Total amount 'gettable' by the agent (to normalize accross different track-lengths)
        """
        super().__init__(weight, AccumulatedDistanceConstant.NAME, clip_min=0, clip_max=10)
        self.current_refline_idx = 0
        self.total_reward = 0
        
    def set_env(self, env):
        super().set_env(env)
        self.reward_per_step = self.total_reward / self.env.reference_line.n_reference_points

    def _get_term(self, observations : dict[str, any], processed_obs : dict[str, any], race_finished : bool, other_terminations : dict[str, bool]):
        next_refline_index, _, _ = self.env.reference_line.get_distance_to_next_point()
        accum_dist_reward = 0
        n_passed = next_refline_index - self.current_refline_idx

        if self.current_refline_idx < next_refline_index: #only give reward if progress in regards to last one was made
            accum_dist_reward = n_passed * self.reward_per_step                
            self.current_refline_idx = next_refline_index

        return accum_dist_reward
    
    def reset(self):
        self.current_refline_idx = 0


class RaceFinishedConstant(RewardTerm):
    """Binary Reward, given if the race is finished; scaled by the amount of reference-line points. This class assumes the max-reference line point length is 50000 points and scales 
    all race-finished evnts according to this max number."""
    NAME = "race_finished"
    
    def __init__(self, weight, max_points : int = 50000):
        """
        Args:
            weight (float) : weight of the reward term used in the sum
            max_points (int)    : Theoretical max amount of reference line points."""
        super().__init__(weight, "race_finished")
        self.max_points = max_points
        self.scale = 1

    def set_env(self, env):
        super().set_env(env)
        assert self.env.reference_line.n_reference_points <= self.max_points, "Cannot scale environment race finished reward term because track has more rerference line points than theoretical max-amount."
        self.scale = self.env.reference_line.n_reference_points / self.max_points

    def _get_term(self, observations, processed_obs, race_finished, other_terminations):
        return race_finished * self.scale

