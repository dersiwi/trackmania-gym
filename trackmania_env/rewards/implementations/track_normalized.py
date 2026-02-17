from typing import Literal

from trackmania_env.rewards.reward_calculation import RewradCalculator, RewardTerm

from trackmania_env.rewards.reward_terms.basic_terms import (
    AccumulatedDistanceReward,
    RaceFinished,
    ConstantRewardTerm,
    SpeedReward,
    LateralDistanceReward,
    TerminationPunishment,
    NoProgressPunishment,
)
from trackmania_env.rewards.reward_terms.tracklength_normalized import AccumulatedDistanceConstant, RaceFinishedConstant

class TrackNormalized(RewradCalculator):
    def __init__(
        self,
        accum_distance_weight: float = 1200,
        race_finished_reward_weight: float = 100,
        other_termination_punishment: float = 20,
        velocity_reward_weight: float = 1,
        distance_to_center_weight: float = 1,
        speed_reward_weight: float = 1,
        lateral_distance_mode: Literal["gauss", "triangle", "trapez"] = "gauss",
        mean: float = 0,
        sigma: float = 1.0,
        yshift: float = -1,
        multiplicator: float = 5,
        dist_scale: float = 0.3,
        normalize: bool = True,
        max_points : int = 50000,
        total_reward_accum_distance = 10,
        **kwargs,
    ):
        """
        Initializes the reward manager with explicit reward weights and parameters
        for modeling centerline distance reward using a Gaussian function.

        Args:
            accum_distance_weight (float):        Weight for accumulated distance reward.
            race_finished_reward_weight (float):  Bonus reward for finishing the race.
            other_termination_punishment (float): Penalty for other termination conditions.
            velocity_reward_weight (float):       Weight for maintaining or increasing velocity.
            distance_to_center_weight (float):    Weight for staying close to the centerline.
            speed_reward_weight (float):          General reward for maintaining speed.
            lateral_distance_mode (str):          Mode used to calculate lateral distance.
        """

        super().__init__(normalize)

        self.accum_distance_weight: float = accum_distance_weight

        self.race_finished_reward_weight: float = race_finished_reward_weight
        self.other_termination_punishment: float = other_termination_punishment
        self.velocity_reward_weight: float = velocity_reward_weight
        self.distance_to_center_weight: float = distance_to_center_weight

        self.speed_reward_weight: float = speed_reward_weight

        self.current_refline_idx:int = 0

        if len(kwargs) > 0:
            print(
                f"Got additional kwargsuments that are not used; ignoring them : {kwargs.keys()}. Maybe they're used by a class that inherits."
            )

        self.terms: list[RewardTerm] = [
            AccumulatedDistanceConstant(accum_distance_weight, total_reward=total_reward_accum_distance),
            RaceFinishedConstant(race_finished_reward_weight, max_points=max_points),
            SpeedReward(speed_reward_weight / SpeedReward.THEORETICAL_MAX_VALUE),
            LateralDistanceReward(distance_to_center_weight, lateral_distance_mode),
            TerminationPunishment((-1) * self.other_termination_punishment),
        ]
    
    def reset(self):
        self.current_refline_idx = 0
        return super().reset()