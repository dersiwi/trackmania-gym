from trackmania_env.rewards.reward_calculation import RewardTerm, RewradCalculator

from trackmania_env.rewards.reward_terms.basic_terms import AccumulatedDistanceReward
from trackmania_env.rewards.reward_terms.penalty_terms import (
    AccumulatedWallPenalty,
    OffTrackPunishment,
    SteeringChangePenalty,
    SteeringHistoryPenalty,
)

class SophyRewards(RewradCalculator):
    def __init__(
        self,
        maxlen_history: int = 3,
        progress_weight: float = 0.014,
        off_course_penalty_weight: float = 0.034,
        wall_penalty_weight: float = 10.0,
        steering_change_penalty_weight: float = 3.0,
        steering_history_penalty_weight: float = 5.0,
        c_d: float = 0.014,
        c_s: float = 182.883569,
        c_o: float = 0.034,
        normalize: bool = False,
        **kwargs,
    ):
        """
        Initializes the GT Sophy-inspired reward function with tunable weights for each component.

        This reward function is based on a weighted sum of progress and penalties that shape
        expert-like racing behavior. It follows the formulation in Wurman et al. (2022), where the
        agent is rewarded for forward progress and penalized for undesirable driving behaviors
        such as going off-course, hitting walls, and erratic steering.

        Args:
            maxlen_history (int, default=3)                     : Number of previous time steps used to compute history-based penalties (e.g., steering consistency).
            progress_weight (float, default=0.014)              : Weight for the course progress reward term , encouraging lap time minimization.
            off_course_penalty_weight (float, default=0.034)    : Weight for the off-course penalty term , discouraging shortcutting or corner cutting.
            wall_penalty_weight (float, default=10.0)           : Weight for the wall-hit penalty term , penalizing contact with track walls.
            steering_change_penalty_weight (float, default=3.0) : Weight for the steering change penalty term , discouraging abrupt steering.
            steering_history_penalty_weight (float, default=5.0): Weight for the steering history penalty term , penalizing inconsistent steering direction over short time spans.
            c_d (float, default=0.014)                          : Threshold steering angle beyond which history-based penalties apply.
            c_s (float, default=182.883569)                     : Sensitivity factor used in the sigmoid function for the steering history penalty.
            c_o (float, default=0.034)                          : Offset in the sigmoid function used in the steering history penalty.

        https://arxiv.org/pdf/2406.12563v1

        Calculates the agent's reward at the current time step, following the reward design
        described in Wurman et al. (2022).

        The reward is formulated as a weighted sum of several components, each targeting
        a specific aspect of safe and efficient racing behavior:

        - Course progress (rp): Encourages forward progress by projecting the agent`s
          position onto the center line of the track and measuring distance advanced.

        - Off-course penalty (ro): Penalizes the agent for having three or more tires outside
          the track boundaries. The penalty is scaled by velocity and the duration spent off-track.

        - Wall penalty (rw): Penalizes collisions with track walls to discourage using walls
          for steering or speed advantages. Also scaled by velocity and contact duration.

        - Steering change penalty (rs): Discourages abrupt steering changes by penalizing
          large differences in steering angles between consecutive time steps.

        - Steering history penalty (rh): Penalizes inconsistent steering patterns within a
          short time window to promote smoother driving behavior. This penalty uses a
          thresholded and smoothed function over recent steering changes.

        The total reward at time t is computed as:
            r_t = w_p * rp_t + w_o * ro_t + w_w * rw_t + w_s * rs_t + w_h * rh_t

        """

        super().__init__(normalize=normalize, **kwargs)

        self.maxlen_history: int = maxlen_history

        self.c_d: float = c_d  # threshold angle
        self.c_s: float = c_s  # sensitivity factor
        self.c_o: float = c_o  # offset value

        self.w_progress: float = progress_weight
        self.w_off_course: float = off_course_penalty_weight
        self.w_wall: float = wall_penalty_weight
        self.w_steer_change: float = steering_change_penalty_weight
        self.w_steer_history: float = steering_history_penalty_weight

        self.terms: list[RewardTerm] = [
            AccumulatedDistanceReward(weight= self.w_progress),
            OffTrackPunishment(weight=self.w_off_course),
            AccumulatedWallPenalty(weight=self.w_wall),
            SteeringChangePenalty(weight=self.w_steer_change),
            SteeringHistoryPenalty(
                weight=self.w_steer_history,
                maxlen_history=self.maxlen_history,
                c_d=self.c_d,
                c_s = self.c_s,
                c_o = self.c_o
            ),
        ]
