from typing import Any
from collections import deque

from trackmania_env.rewards.reward_calculation import RewardTerm, BoundedRewardterm
from trackmania_env.utils import constants
from game_interaction.ipc_fields import IPCFields
from tminterface.structs import SimStateData
import numpy as np


class OffTrackPunishment(RewardTerm):
    """Implements a punishment if the agent is off-track."""

    # Theroretical maximial Value of the reward-term, before weighting.
    THEORETICAL_MAX_VALUE: float = 0.0
    # Theroretical minimal Value of the reward-term, before weighting.
    THEORETICAL_MIN_VALUE: float = -1.0
    NAME: str = "off_track"

    def __init__(
        self,
        weight: float,
        clip_min: float = THEORETICAL_MIN_VALUE,
        clip_max: float = THEORETICAL_MAX_VALUE,
    ):
        super().__init__(
            weight, OffTrackPunishment.NAME, clip_min=clip_min, clip_max=clip_max
        )
        self.total_off_course_time: float = 0
        self.last_time: float = 0

    def _get_term(
        self,
        observations: dict[str, Any | SimStateData],
        processed_obs: dict[str, Any],
        race_finished: bool,
        other_terminations: dict[str, bool],
    ):
        """
        Calculates a penalty if the agent is off-course, based on distance and elevation
        from the reference line. The penalty increases the longer the agent stays off-course.
        """

        sim_state: SimStateData = observations[IPCFields.SIMSTATE]
        current_time = sim_state.time / constants.MILLISECONDS_TO_SECONDS
        delta_time = current_time - self.last_time

        # Get distance and height difference to the reference line
        next_index, lateral_distance, _ = (
            self.env.reference_line.get_distance_to_next_point()
        )
        refline_height: float = self.env.reference_line.reference_line[next_index][1]
        agent_height: float = sim_state.position[1]
        height_difference: float = abs(agent_height - refline_height)

        # Determine if the agent is off-course
        is_off_course = (
            lateral_distance >= constants.MAX_DISTANCE_TO_REFLINE
            or height_difference >= constants.MAX_HEIGHT_DIFERENCE
        )

        reward = 0

        if is_off_course:
            # Accumulate off-course time and apply penalty
            self.total_off_course_time += delta_time
            reward = -self.total_off_course_time
        else:
            # Reset timer if agent is back on course
            self.total_off_course_time = 0
        return reward

    def reset(self):
        self.last_time = 0
        self.total_off_course_time = 0

class HasWallConatact(BoundedRewardterm):

    def __init__(self, weight):
        """Boolean term, returns if 1, if one of the tires has wall contact."""
        super().__init__(weight, "wall_contact")

    def _get_term(self, observations, processed_obs, race_finished, other_terminations):
        #ssD : SimStateData = observations[IPCFields.SIMSTATE]
        #svc : SceneVehicleCar = ssD.scene_mobil 
        return observations[IPCFields.SIMSTATE].scene_mobil.has_any_lateral_contact

class AccumulatedWallPenalty(RewardTerm):
    # Theroretical maximial Value of the reward-term, before weighting.
    THEORETICAL_MAX_VALUE: float = 0
    # Theroretical minimal Value of the reward-term, before weighting.
    THEORETICAL_MIN_VALUE: float = -np.inf
    NAME: str = "accumulated_wall_penalty"

    def __init__(
        self,
        weight: float,
        min_wall_contact_time: float = 0.5,
        clip_min: float = -2.0,
        clip_max: float = 0.0,
    ):
        super().__init__(
            weight, AccumulatedWallPenalty.NAME, clip_min=clip_min, clip_max=clip_max
        )

        self.min_wall_contact_time: float = min_wall_contact_time
        self.last_time: float = 0

    def _get_term(
        self,
        observations: dict[str, Any | SimStateData],
        processed_obs: dict[str, Any],
        race_finished: bool,
        other_terminations: dict[str, bool],
    ):
        """
        Computes a penalty based on how long the agent maintains lateral contact with the wall.
        Penalty is applied only if contact exceeds a predefined threshold.
        """

        sim_state: SimStateData = observations[IPCFields.SIMSTATE]
        current_time = sim_state.time / constants.MILLISECONDS_TO_SECONDS
        delta_time = current_time - self.last_time

        reward = 0

        if sim_state.scene_mobil.has_any_lateral_contact:
            # Accumulate contact time
            self.continuous_wall_contact_time += delta_time

            # Apply penalty only if contact exceeds threshold
            if self.continuous_wall_contact_time > self.min_wall_contact_time:
                # Penalty is based only on the excess contact time beyond the threshold
                excess_contact_time = (
                    self.continuous_wall_contact_time - self.min_wall_contact_time
                )
                reward = -excess_contact_time
        else:
            # Reset timer when contact is lost
            self.continuous_wall_contact_time = 0.0

        return reward

    def reset(self):
        self.last_time = 0
        self.continuous_wall_contact_time = 0.0


class SteeringChangePenalty(RewardTerm):
    THEORETICAL_MAX_VALUE: float = 0
    THEORETICAL_MIN_VALUE: float = -np.inf
    NAME: str = "steering_change_penalty"

    def __init__(
        self,
        weight: float,
        clip_min: float = -2.0,
        clip_max: float = 0.0,
    ):
        super().__init__(
            weight, SteeringChangePenalty.NAME, clip_min=clip_min, clip_max=clip_max
        )

        self.previous_steering_angle: float = 0.0

    def _get_term(
        self,
        observations: dict[str, Any | SimStateData],
        processed_obs: dict[str, Any],
        race_finished: bool,
        other_terminations: dict[str, bool],
    ):
        ssD: SimStateData = observations[IPCFields.SIMSTATE]
        turning_rate: float = ssD.scene_mobil.turning_rate

        steering_angle: float = constants.MAX_DEGREE_RADIANS * turning_rate

        # Safety check
        assert (
            abs(steering_angle) <= constants.MAX_DEGREE_RADIANS + constants.EPSILON
        ), f"steering_angle out of range: {steering_angle}"

        steering_change = abs(steering_angle - self.previous_steering_angle)

        self.previous_steering_angle = steering_angle

        rs: float = -steering_change

        return rs


class SteeringHistoryPenalty(RewardTerm):
    THEORETICAL_MAX_VALUE: float = 0
    THEORETICAL_MIN_VALUE: float = -np.inf
    NAME: str = "steering_history_penalty"

    def __init__(
        self,
        weight: float,
        maxlen_history: int = 3,
        clip_min: float = -2.0,
        clip_max: float = 0.0,
        c_d: float = 0.014,
        c_s: float = 182.883569,
        c_o: float = 0.034,
    ):
        super().__init__(
            weight, SteeringHistoryPenalty.NAME, clip_min=clip_min, clip_max=clip_max
        )
        self.maxlen_history: int = maxlen_history
        self.c_d: float = c_d
        self.c_s: float = c_s
        self.c_o: float = c_o

        self.angles = deque([0.0] * self.maxlen_history, maxlen=self.maxlen_history)

    def _get_term(
        self,
        observations: dict[str, Any],
        processed_obs: dict[str, Any],
        race_finished: bool,
        other_terminations: dict[str, bool],
    ):
        ssD: SimStateData = observations[IPCFields.SIMSTATE]
        turning_rate: float = ssD.scene_mobil.turning_rate

        steering_angle: float = constants.MAX_DEGREE_RADIANS * turning_rate

        # Safety check
        assert (
            abs(steering_angle) <= constants.MAX_DEGREE_RADIANS + constants.EPSILON
        ), f"steering_angle out of range: {steering_angle}"

        self.angles.append(steering_angle)

        h_a_t = np.array(self.angles, dtype=np.float32)

        h_d_t = np.diff(h_a_t)

        h_d_t_tail = h_d_t[-2:]

        has_sign_flip: bool = np.sign(h_d_t[-1]) != np.sign(h_d_t[-2])
        above_threshold: bool = np.all(np.abs(h_d_t_tail) > self.c_d)

        # Indicator for inconsistent steering behavior (m_t)
        m_t: float = 1.0 if (has_sign_flip and above_threshold) else 0.0

        # Calculate sum of absolute deltas
        delta_t: float = np.sum(np.abs(h_d_t_tail))

        # rh = -m_t / (1 + exp(-delta_t))
        penalty: float = -m_t / (1.0 + np.exp(-self.c_s* (delta_t- self.c_o)))

        return penalty
