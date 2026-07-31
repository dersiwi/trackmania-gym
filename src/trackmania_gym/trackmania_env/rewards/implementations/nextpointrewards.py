from __future__ import annotations
from typing import Literal

from trackmania_gym.trackmania_env.rewards.reward_calculation import RewradCalculator, RewardTerm

from trackmania_gym.trackmania_env.rewards.reward_terms.basic_terms import (
    AccumulatedDistanceReward,
    RaceFinished,
    ConstantRewardTerm,
    SpeedReward,
    LateralDistanceReward,
    TerminationPunishment,
    NoProgressPunishment,
)
from trackmania_gym.trackmania_env.rewards.reward_terms.penalty_terms import HasWallConatact
class NextPointRewards(RewradCalculator):
    def __init__(
        self,
        accum_distance_weight: float = 1200,
        race_not_finished_weight: float = 0.05,
        race_finished_reward_weight: float = 100,
        other_termination_punishment: float = 20,
        velocity_reward_weight: float = 1,
        backward_weight: float = 0,
        distance_to_center_weight: float = 1,
        velocity_change_reward_weight: float = 30,
        speed_reward_weight: float = 1,
        lateral_distance_mode: Literal["gauss", "triangle", "trapez"] = "gauss",
        mean: float = 0,
        sigma: float = 1.0,
        yshift: float = -1,
        multiplicator: float = 5,
        dist_scale: float = 0.3,
        normalize: bool = False,
        **kwargs,
    ):
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

        self.accum_distance_weight: float = accum_distance_weight
        self.race_not_finished_weight: float = race_not_finished_weight

        self.race_finished_reward_weight: float = race_finished_reward_weight
        self.other_termination_punishment: float = other_termination_punishment
        self.velocity_reward_weight: float = velocity_reward_weight
        self.backward_weight: float = backward_weight
        self.distance_to_center_weight: float = distance_to_center_weight
        self.velocity_change_reward_weight: float = velocity_change_reward_weight

        self.speed_reward_weight: float = speed_reward_weight

        self.current_refline_idx:int = 0

        if len(kwargs) > 0:
            print(
                f"Got additional kwargsuments that are not used; ignoring them : {kwargs.keys()}. Maybe they're used by a class that inherits."
            )

        self.terms: list[RewardTerm] = [
            AccumulatedDistanceReward(accum_distance_weight),
            RaceFinished(race_finished_reward_weight),
            ConstantRewardTerm((-1) * race_not_finished_weight),
            SpeedReward(speed_reward_weight / SpeedReward.THEORETICAL_MAX_VALUE),
            LateralDistanceReward(distance_to_center_weight, lateral_distance_mode),
            TerminationPunishment((-1) * self.other_termination_punishment),
        ]
    
    def reset(self):
        self.current_refline_idx = 0
        return super().reset()


class OptimizeRaceTiem(RewradCalculator):
    def __init__(
        self,
        accum_distance_weight:float=1200,
        race_not_finished_weight:float=0.05,
        race_finished_reward_weight:float=100,
        other_termination_punishment:float=20,
        speed_reward_weight:float=1,
        wall_contact:float=-0.2,
        normalize:bool=False,
        **kwargs,
    ):
        super().__init__(normalize, **kwargs)
        self.terms: list[RewardTerm] = [
            AccumulatedDistanceReward(accum_distance_weight),
            RaceFinished(race_finished_reward_weight, scaled_by_steps_taken=True),
            ConstantRewardTerm((-1) * race_not_finished_weight),
            SpeedReward(speed_reward_weight / SpeedReward.THEORETICAL_MAX_VALUE),
            TerminationPunishment((-1) * other_termination_punishment),
            HasWallConatact(wall_contact),
        ]


class RaceFinishedRewards(NextPointRewards):
    """Similar to NextPointRewards, but it scales the distance to center in relation to the accumulated distance."""

    def __init__(
        self,
        steps_without_progress_until_punishment: int,
        use_punishment: bool,
        normalize: bool = False,
        **kwargs,
    ):
        """
        Args:
            steps_without_progress_until_punishment (int) : Amount of steps the agent can do without achieving any progress (progress is measured by accumulated distance)
                until the reward manager punishes the agent
            use_punishment (bool)   : If False, it never punishes the agent for not making any progress. If True, it uses steps_without_progress_until_punishment.
            noramlize (bool)        : Propagated to parent-class (IF set, normalizes returns using running mean)
        """
        super().__init__(normalize=normalize, **kwargs)

        self.terms: list[RewardTerm] = [
            RaceFinished(self.race_finished_reward_weight, scaled_by_steps_taken=True),
            AccumulatedDistanceReward(
                self.accum_distance_weight,
                enhanced_by_amount_travelled=True,
                exponential_factor=1.2,
            ),
            TerminationPunishment((-1) * self.other_termination_punishment),
            HasWallConatact(-0.15),
        ]

        if use_punishment:
            self.terms.append(
                NoProgressPunishment(
                    self.race_finished_reward_weight,
                    steps_without_progress_until_punishment,
                )
            )
