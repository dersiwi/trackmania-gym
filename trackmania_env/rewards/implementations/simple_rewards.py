from __future__ import annotations

from trackmania_env.rewards.reward_calculation import RewradCalculator, RewardTerm

from trackmania_env.rewards.reward_terms.basic_terms import (
    AccumulatedDistanceReward,
    RaceFinished,
    TerminationPunishment,
    NoProgressPunishment,
)
from trackmania_env.rewards.reward_terms.penalty_terms import HasWallConatact


class RaceFinishedRewards(RewradCalculator):
    """Similar to NextPointRewards, but it scales the distance to center in relation to the accumulated distance."""

    def __init__(
        self,
        race_finished_reward_weight: float,
        scaled_by_steps_taken: bool,
        accum_distance_weight: float,
        accum_enhanced_by_amount_travelled: bool,
        accum_exp_factor: float,
        other_termination_punishment: float,
        wall_contact_weight: float,
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
            RaceFinished(race_finished_reward_weight, scaled_by_steps_taken=scaled_by_steps_taken),
            AccumulatedDistanceReward(
                accum_distance_weight,
                enhanced_by_amount_travelled=accum_enhanced_by_amount_travelled,
                exponential_factor=accum_exp_factor,
            ),
            TerminationPunishment(other_termination_punishment),
            HasWallConatact(wall_contact_weight),
        ]

        if use_punishment:
            self.terms.append(
                NoProgressPunishment(
                    race_finished_reward_weight,
                    steps_without_progress_until_punishment,
                )
            )
