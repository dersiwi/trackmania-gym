from __future__ import annotations
from trackmania_env.utils.position_buffer import PositionBuffer
from tminterface.structs import CheckpointData, SimStateData, CheckpointTime
from game_interaction.ipc_fields import IPCFields
import numpy as np
from numba import jit

from trackmania_env.rewards.reward_calculation import RewradCalculator
from trackmania_env.utils.reference_line_manager import ReferenceLineManager

import logging
from configs.config import RewardManagerCfg

class NextPointRewards(RewradCalculator):

    def __init__(self, reward_cfg : RewardManagerCfg):
        super().__init__(reward_cfg)
        #self.refline_manager = ReferenceLineManager(filepath_referenceline)

        self.accum_distance_weight = reward_cfg.rewardterm_weights["accum_distance_wegiht"]
        self.race_not_finished_weight = reward_cfg.rewardterm_weights["race_not_finished_weight"]
        
        self.race_finished_reward = reward_cfg.rewardterm_weights["race_finished_reward"]
        self.other_termination_punishment = reward_cfg.rewardterm_weights["other_termination_punishment"]
        self.velocity_reward_weight = reward_cfg.rewardterm_weights["velocity_reward_weight"]
        self.backward_weight = reward_cfg.rewardterm_weights["backward_weight"]
        self.distance_to_center_weight = reward_cfg.rewardterm_weights["distance_to_center_weight"]
        self.velocity_change_reward_weight = reward_cfg.rewardterm_weights["velocity_change_reward_weight"]

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logging_freq = 1000
        
        self.current_refline_idx : int = 0

        self.max_lateral_difference = 12 # maximal lateral difference, this is an estimate.

        self.last_simstate : SimStateData = None

    def _get_normed_velocity(self, velocity : list[float, float, float]) -> float:
        return np.linalg.norm(np.array(velocity) / 1000) #1000 == max velocity

    def calculate_reward(self, observations, processed_obs : dict[str, any], race_finished, other_terminations):
        ssD : SimStateData = observations[IPCFields.SIMSTATE]
        reward = 0
        
        next_refline_index, d, drel = self.refline_manager.get_distance_to_next_point()
        accum_dist_reward = 0
        if not self.current_refline_idx == next_refline_index: #only give reward if progress in regards to last one was made
            accum_dist_reward = self.refline_manager.get_discrete_distance(next_refline_index) * self.accum_distance_weight
            self.current_refline_idx = next_refline_index

        current_velocity_normed = self._get_normed_velocity(ssD.velocity)
        velocity_change_reward = 0
        if not self.last_simstate == None:
            velocity_change_reward = (current_velocity_normed - self._get_normed_velocity(self.last_simstate.velocity)) * self.velocity_change_reward_weight

        distance_to_center_reward = 0
        if next_refline_index > 0:
            # only calculate reward to centerline once car is within the firsrt linesgement, 
            # e.g. if the agent drives backwards out of map immediately he will always get max reward, because this term will always be 1/12 * self.ditsance_to_center_weight
            distance_to_center_reward = self.distance_to_center_weight * (1 - np.clip(self.refline_manager.calculate_lateral_difference(idx = next_refline_index, car_position=ssD.position), 
                                                    a_min = 0, a_max = self.max_lateral_difference) / self.max_lateral_difference) # inverse the distance, such that reward is bigger once distance gets smaller
            
        #velocity_reward = self.velocity_reward_weight * current_velocity_normed
        race_not_finished_reward = (-1) * self.race_not_finished_weight
        race_finished = race_finished * self.race_finished_reward
        other_term_reward = (-1) * other_terminations * self.other_termination_punishment
        backward_punishment = (-1) * np.clip(d, a_min=0, a_max=100) / 100 * self.backward_weight
        
        reward = accum_dist_reward + race_not_finished_reward + race_finished + other_term_reward + backward_punishment + distance_to_center_reward + velocity_change_reward
        self.last_simstate = ssD
        return reward, {"total" : reward, 
                        "accumulated_distance" : accum_dist_reward,
                        "distance_to_center" : distance_to_center_reward,
                        "nextpoint_reference_index" : next_refline_index,
                        "race_not_finished":race_not_finished_reward,
                        "race_finished" : race_finished,
                        "other_terminations":other_term_reward,
                        "backward_punishment" : backward_punishment,
                        "velocity_change_reward" : velocity_change_reward}




    def reset(self):
        self.current_refline_idx
        return super().reset()