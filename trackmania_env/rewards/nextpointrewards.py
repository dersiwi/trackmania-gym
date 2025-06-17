from __future__ import annotations
from trackmania_env.utils.position_buffer import PositionBuffer
from tminterface.structs import CheckpointData, SimStateData, CheckpointTime
from game_interaction.ipc_fields import IPCFields
import numpy as np
from numba import jit

from trackmania_env.rewards.reward_calculation import RewradCalculator
from trackmania_env.utils.reference_line_manager import ReferenceLineManager

import logging

class NextPointRewards(RewradCalculator):

    def __init__(self, filepath_referenceline : str):
        super().__init__()
        self.refline_manager = ReferenceLineManager(filepath_referenceline)

        self.accum_distance_weight = 5
        self.race_not_finished_weight = 0.05
        self.race_finished_reward = 100
        self.other_termination_punishment = 2
        self.velocity_reward_weight = 2
        self.backward_weight = 5

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logging_freq = 1000
        
        self.current_refline_idx : int = 0


    def calculate_reward(self, observations, race_finished, other_terminations):
        ssD : SimStateData = observations[IPCFields.SIMSTATE]
        reward = 0
        
        next_refline_index, d, drel = self.refline_manager.get_distance_to_next_point(ssD.position)
        accum_dist_reward = 0
        if not self.current_refline_idx == next_refline_index: #only give reward if progress in regards to last one was made
            accum_dist_reward = self.accum_distance_weight * drel
            self.current_refline_idx = next_refline_index
        
        velocity_reward = self.velocity_reward_weight * np.linalg.norm(np.array(ssD.velocity) / 1000) #1000 == max velocity
        race_not_finished_reward = (-1) * self.race_not_finished_weight
        race_finished = race_finished * self.race_finished_reward
        other_term_reward = (-1) * other_terminations * self.other_termination_punishment
        backward_punishment = (-1) * np.clip(d, a_min=0, a_max=100) / 100 * self.backward_weight
        
        reward = accum_dist_reward + race_not_finished_reward + race_finished + other_term_reward + velocity_reward + backward_punishment

        return reward, {"total" : reward, 
                        "accumulated_distance" : accum_dist_reward,
                        "nextpoint_reference_index" : next_refline_index,
                        "race_not_finished":race_not_finished_reward,
                        "race_finished" : race_finished,
                        "other_terminations":other_term_reward,
                        "vel_reward":velocity_reward,
                        "backward_punishment" : backward_punishment}




    def reset(self):
        self.refline_manager.reset()
        self.current_refline_idx
        return super().reset()