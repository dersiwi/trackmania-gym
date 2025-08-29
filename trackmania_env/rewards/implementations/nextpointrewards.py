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
from trackmania_env.rewards.terms import RaceFinished, ConstantRewardTerm, LateralDistanceReward, AccumulatedDistanceReward, SpeedReward, NoProgressPunishment, TerminationPunishment


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
        accum_distance_weight=1200,
        race_not_finished_weight=0.05,
        race_finished_reward_weight=100,
        other_termination_punishment=20,
        velocity_reward_weight=1,
        backward_weight=0,
        distance_to_center_weight=1,
        velocity_change_reward_weight=30,
        speed_reward_weight=1,
        lateral_distance_mode="gauss",
        mean=0,
        sigma=1,
        yshift=-1,
        multiplicator=5,
        dist_scale=0.3,
        normalize=False,
        off_course_penalty_weight:float = 0.034, 
        wall_penalty_weight:float = 10.0,
        min_wall_contact_time:float = 0.5,   
        minimum_punishment_speed:float = 10.0,
        **kwargs
    ):
        super().__init__(
            accum_distance_weight,
            race_not_finished_weight,
            race_finished_reward_weight,
            other_termination_punishment,
            velocity_reward_weight,
            backward_weight,
            distance_to_center_weight,
            velocity_change_reward_weight,
            speed_reward_weight,
            lateral_distance_mode,
            mean,
            sigma,
            yshift,
            multiplicator,
            dist_scale,
            normalize,
            **kwargs
        )
        self.last_time = 0
        self.continuous_wall_contact_time = 0
        self.total_off_course_time = 0

        self.min_wall_contact_time = min_wall_contact_time
        self.minimum_punishment_speed = minimum_punishment_speed

        self.w_off_course = off_course_penalty_weight
        self.w_wall = wall_penalty_weight
    
    def reset(self):
        super().reset()
        self.last_time = 0
        self.continuous_wall_contact_time = 0.0
        self.total_off_course_time = 0



    def get_sum_of_weighted_rewards(self, observations, processed_obs, race_finished, other_terminations):      
        reward,info = super().get_sum_of_weighted_rewards(observations, processed_obs, race_finished, other_terminations)

        ssD : SimStateData = observations[IPCFields.SIMSTATE]
        time =  ssD.time/ constants.MILLISECONDS_TO_SECONDS
        delta_time = time - self.last_time
        next_refline_index , d, _ = self.refline_manager.get_distance_to_next_point()
        speed = ssD.display_speed / constants.MS_TO_KMH
        refline_z_coord = self.refline_manager.reference_line[next_refline_index][1]
        agent_z_coord = ssD.position[1]
        height_diff = abs(agent_z_coord-refline_z_coord)
        
        # off-course
        lateral_distance = d
        is_off_course = lateral_distance >= constants.MAX_DISTANCE_TO_REFLINE or height_diff >= constants.MAX_HEIGHT_DIFERENCE 
        ro = 0

        if is_off_course:
            # Increment the total off-course time
            self.total_off_course_time += delta_time
            # The penalty is based on the new time accumulated in this step
            ro = -self.total_off_course_time * self.w_off_course
        else:
            self.total_off_course_time = 0
            
        ro = max(ro,-20.0)
        info["off-course"] = ro

        # wall
        rw = 0
        if ssD.scene_mobil.has_any_lateral_contact:
            # increment timer by the time elapsed
            self.continuous_wall_contact_time += delta_time
            
            # Punish only if contact time exceeds threshold (to allow things like speedbumps)
            if self.continuous_wall_contact_time > self.min_wall_contact_time:
                # The punishment starts small and grows as the agent stays on the wall
                punishment_duration = self.continuous_wall_contact_time - self.min_wall_contact_time
                rw = -punishment_duration #* max(speed,self.minimum_punishment_speed) # use minimum speed to ensure a penalty even when not moving
                rw *= self.w_wall 
        else: 
            self.continuous_wall_contact_time = 0.0 # Reset timer if contact is lost
        info["wall"] = rw

        rw = max(rw,-20.0)

        reward += rw+ro
        self.last_time = time 
        return super().normalize_reward(reward),info

