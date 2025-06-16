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
        self.velocity_reward_weight = 1

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logging_freq = 1000
        self.step = 0


    def calculate_reward(self, observations, race_finished, other_terminations):
        ssD : SimStateData = observations[IPCFields.SIMSTATE]
        reward = 0
        
        i, d, drel = self.refline_manager.get_distance_to_next_point(ssD.position)
        velocity_reward = self.velocity_reward_weight * np.linalg.norm(np.array(ssD.velocity) / 1000) #1000 == max velocity
        accum_dist_reward = self.accum_distance_weight * drel
        race_not_finished_reward = (-1) * self.race_not_finished_weight
        race_finished = race_finished * self.race_finished_reward
        other_term_reward = (-1) * other_terminations * self.other_termination_punishment
        
        reward = accum_dist_reward + race_not_finished_reward + race_finished + other_term_reward + velocity_reward
        if self.step % self.logging_freq == 0:
            self.logger.info(f"Calculated_Rewards;{reward}:accumulated_distance;{accum_dist_reward};{i};{d}:race_not_finished;{race_not_finished_reward}:race_finished;{race_finished}:other_terminations;{other_term_reward}:vel_reward;{velocity_reward}")
        self.step += 1
        return reward




    def reset(self):
        self.refline_manager.reset()
        self.step = 0
        return super().reset()