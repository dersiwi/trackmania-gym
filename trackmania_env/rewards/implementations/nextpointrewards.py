from __future__ import annotations
from tminterface.structs import CheckpointData, SimStateData, CheckpointTime
import numpy as np
from scipy.stats import norm
import logging
from typing import Literal

from trackmania_env.rewards.reward_calculation import RewradCalculator
from game_interaction.ipc_fields import IPCFields
from trackmania_env.utils import constants
from trackmania_env.utils.lateral_distance_manager import LateralDistanceManager

from trackmania_env.rewards.terms import (
    RaceFinished,
    ConstantRewardTerm,
    LateralDistanceReward,
    AccumulatedDistanceReward,
    SpeedReward,
    NoProgressPunishment,
    TerminationPunishment,
    OffTrackPunishment,
    AccumulatedWallPenalty)

from trackmania_env.rewards.implementations.advanced_skills_rewards import DriftReward,AirBrakeReward,SpeedSlideReward

class NextPointRewards(RewradCalculator):

    def __init__(self, 
                accum_distance_weight: float=1200,
                race_not_finished_weight: float=0.05,
                race_finished_reward_weight: float=100,
                other_termination_punishment: float=20,
                velocity_reward_weight: float=1,
                backward_weight: float=0,
                distance_to_center_weight: float=1,
                velocity_change_reward_weight: float=30,
                speed_reward_weight: float=1,
                lateral_distance_mode: Literal["gauss", "triangle", "trapez"] = "gauss",
                mean: float = 0,
                sigma: float = 1.0,
                yshift: float = -1,
                multiplicator: float = 5,
                dist_scale: float = 0.3, 
                normalize : bool = False,
                **kwargs ):
        """
        Initializes the reward manager with explicit reward weights and parameters
        for modeling centerline distance reward using a Gaussian function.

        Args:
            accum_distance_weight (float):        Weight for accumulated distance reward.
            race_not_finished_weight (float):     Penalty weight for not finishing the race.
            race_finished_reward_weight (float):  Bonus reward for finishing the race.
            other_termination_punishment (float): Penalty for other termination conditions.
            velocity_reward_weight (float):       Weight for maintaining or increasing velocity.
            backward_weight (float): Penalty      weight for moving backward.
            distance_to_center_weight (float):    Weight for staying close to the centerline.
            velocity_change_reward_weight (float):Weight for smooth velocity changes.
            speed_reward_weight (float):          General reward for maintaining speed.
            lateral_distance_mode (str):          Mode used to calculate lateral distance.
    """
        
        super().__init__(normalize)

        self.accum_distance_weight = accum_distance_weight
        self.race_not_finished_weight = race_not_finished_weight
        
        self.race_finished_reward_weight = race_finished_reward_weight
        self.other_termination_punishment = other_termination_punishment
        self.velocity_reward_weight = velocity_reward_weight
        self.backward_weight = backward_weight
        self.distance_to_center_weight = distance_to_center_weight
        self.velocity_change_reward_weight = velocity_change_reward_weight

        self.speed_reward_weight = speed_reward_weight


        if len(kwargs) > 0:
            print(f"Got additional kwargsuments that are not used; ignoring them : {kwargs.keys()}. Maybe they're used by a class that inherits.")

        self.reward_terms = [AccumulatedDistanceReward(accum_distance_weight),
                             RaceFinished(race_finished_reward_weight),
                             ConstantRewardTerm((-1) * race_not_finished_weight),
                             SpeedReward(speed_reward_weight / 1000),
                             LateralDistanceReward(distance_to_center_weight, lateral_distance_mode),
                             TerminationPunishment(self.other_termination_punishment)
]
        
    def reset(self):
        self.current_refline_idx = 0
        return super().reset()
    


class RaceFinishedRewards(NextPointRewards):
    """Similar to NextPointRewards, but it scales the distance to center in relation to the accumulated distance."""
    def __init__(self, steps_without_progress_until_punishment : int, use_punishment : bool, normalize : bool = False, **kwargs):
        """
        Args:
            steps_without_progress_until_punishment (int) : Amount of steps the agent can do without achieving any progress (progress is measured by accumulated distance)
                until the reward manager punishes the agent
            use_punishment (bool)   : If False, it never punishes the agent for not making any progress. If True, it uses steps_without_progress_until_punishment.
            noramlize (bool)        : Propagated to parent-class (IF set, normalizes returns using running mean)
            """
        super().__init__(normalize=normalize, **kwargs)

        self.reward_terms = [
            RaceFinished(self.race_finished_reward_weight, scaled_by_steps_taken=True),
            AccumulatedDistanceReward(self.accum_distance_weight),
            TerminationPunishment(self.other_termination_punishment)]
        
        if use_punishment:
            self.reward_terms.append(NoProgressPunishment(self.race_finished_reward_weight, steps_without_progress_until_punishment))

class NextPointRewards3(NextPointRewards):

    def __init__(
        self,
        off_course_penalty_weight:float = 0.034, 
        wall_penalty_weight:float = 10.0,
        min_wall_contact_time:float = 0.5,   
        normalize=False,
        **kwargs
    ):
        super().__init__(normalize=normalize, **kwargs)
        self.reward_terms.append(OffTrackPunishment(weight=off_course_penalty_weight))
        self.reward_terms.append(AccumulatedWallPenalty(weight=wall_penalty_weight,min_wall_contact_time = min_wall_contact_time))

class NextPointDriftReward(NextPointRewards):
    def __init__(self, 
                drift_reward_weight = 1,
                normalize = False,
                **kwargs):
        super().__init__(normalize=normalize, **kwargs)
        self.reward_terms.append(DriftReward(drift_reward_weight))

class AirBrakeNextPointReward(NextPointRewards):

    def __init__(self, 
                air_brake_reward_weight = 1,
                normalize = False,
                **kwargs):
        super().__init__(normalize=normalize, **kwargs)
        self.reward_terms.append(AirBrakeReward(air_brake_reward_weight))        


class SpeedSplideNextPointReward(NextPointRewards):
   
    def __init__(self, 
                speed_slide_reward_weight = 1,
                normalize = False,
                **kwargs):
        super().__init__(normalize=normalize, **kwargs)
        self.reward_terms.append(SpeedSlideReward(speed_slide_reward_weight))        