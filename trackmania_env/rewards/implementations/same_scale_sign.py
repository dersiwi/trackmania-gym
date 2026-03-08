from __future__ import annotations

from game_interaction.ipc_fields import IPCFields
from trackmania_env.rewards.reward_calculation import RewradCalculator, RewardTerm, BoundedRewardterm

from trackmania_env.rewards.reward_terms.basic_terms import (
    AccumulatedDistanceReward,
    RaceFinished,
)


class PosTerminationPunishment(BoundedRewardterm):
    """Rewards the agent if he does not terminate the episode."""

    def __init__(self, weight):
        super().__init__(weight, "terminations")

    def _get_term(self, observations, processed_obs, race_finished, other_terminations):
        ot = False
        if "stuck" in other_terminations:
            ot = other_terminations["stuck"]
        if "no_progress" in other_terminations:
            ot = ot or other_terminations["no_progress"]

        other_term_reward = 1 - int(ot)
        return other_term_reward


class PosHasWallConatact(BoundedRewardterm):
    def __init__(self, weight):
        """Boolean term, returns if 1 agent has NO wall contact otherwise zero."""
        super().__init__(weight, "wall_contact")

    def _get_term(self, observations, processed_obs, race_finished, other_terminations):
        return 1 - int(observations[IPCFields.SIMSTATE].scene_mobil.has_any_lateral_contact)


# Same scale and sign (SSS) race finished
class SSS_RaceFinishedRewards(RewradCalculator):
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
            steps_without_progress_until_punishment (int) : Amount of steps the agent can do without achieving any progress
            (progress is measured by accumulated distance) until the reward manager punishes the agent
            use_punishment (bool)   : If False, it never punishes the agent for not making any progress. If True, it uses
            steps_without_progress_until_punishment.
            noramlize (bool)        : Propagated to parent-class (IF set, normalizes returns using running mean)
        """
        super().__init__(normalize=normalize, **kwargs)

        self.terms: list[RewardTerm] = [  # TODO : Revert!!!!
            RaceFinished(race_finished_reward_weight, scaled_by_steps_taken=scaled_by_steps_taken),
            AccumulatedDistanceReward(
                accum_distance_weight,
                enhanced_by_amount_travelled=accum_enhanced_by_amount_travelled,
                exponential_factor=accum_exp_factor,
            ),
            PosTerminationPunishment(other_termination_punishment),
            PosHasWallConatact(wall_contact_weight),
        ]
