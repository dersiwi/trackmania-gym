"""
Contains basic reward terms
    - AccumulatedDistanceReward
    - LateralDistanceReward
    - TerminationPunishment
    - HasWallContact
    - SpeedReward
    - RaceFinished
    - ConstantRewardTerm
    - NoProgressPunishment
    - OffTrackPunishment
    - AccumulatedWallPenalty
    """

from trackmania_env.rewards.reward_calculation import RewardTerm, BoundedRewardterm
from trackmania_env.utils import constants
from trackmania_env.utils.lateral_distance_manager import LateralDistanceManager
from game_interaction.ipc_fields import IPCFields
from tminterface.structs import SimStateData, SceneVehicleCar
import numpy as np
class AccumulatedDistanceReward(RewardTerm):
    """Calculates the reward along indicating the distance driven along the centerline"""

    NAME = "accumulated_distance"

    def __init__(self, weight, enhanced_by_amount_travelled : bool = False, exponential_factor : float = 1.0):
        """Iniitialites
        Args:
            weight (float)  : Weight of the term
            enhanced_by_amount_travelled (bool) : If true, multiplies the reward by the amount of refline-points passed (to give more incentive to pass more at a single environment step)
            exponential_factor (float)          : Number of steps travelled in this step is raised to this power. Default 1.
        """
        super().__init__(weight, AccumulatedDistanceReward.NAME, clip_min=0, clip_max=10)
        self.current_refline_idx = 0
        self.enhanced_by_amount_travelled = enhanced_by_amount_travelled
        self.exponential_factor = exponential_factor

    def _get_term(self, observations : dict[str, any], processed_obs : dict[str, any], race_finished : bool, other_terminations : dict[str, bool]):
        next_refline_index, _, _ = self.env.reference_line.get_distance_to_next_point()
        accum_dist_reward = 0
        n_passed = next_refline_index - self.current_refline_idx
        if self.current_refline_idx < next_refline_index: #only give reward if progress in regards to last one was made

            for i in range(n_passed):
                accum_dist_reward += self.env.reference_line.get_discrete_distance(self.current_refline_idx + i)
                
            self.current_refline_idx = next_refline_index
        if self.enhanced_by_amount_travelled:
            accum_dist_reward = accum_dist_reward * n_passed ** self.exponential_factor
        return accum_dist_reward
    
    def reset(self):
        self.current_refline_idx = 0
    
class LateralDistanceReward(RewardTerm):
    THEORETICAL_MAX_VALUE = np.inf
    """Theroretical maximial Value of the reward-term, before weighting."""
    THEORETICAL_MIN_VALUE = 0
    """Theroretical minimal Value of the reward-term, before weighting."""

    NAME = "lateral_distance"

    def __init__(self, weight, lateral_distance_mode : str, clip_min = 0.0, clip_max = 1.0):
        super().__init__(weight, LateralDistanceReward.NAME, clip_min, clip_max)
        self.lateral_distance_manager = LateralDistanceManager.get_instance(lateral_distance_mode)

    def _get_term(self, observations : dict[str, any], processed_obs : dict[str, any], race_finished : bool, other_terminations : dict[str, bool]):
        next_refline_index, _, _ = self.env.reference_line.get_distance_to_next_point()
        if next_refline_index <= 0:
                # only calculate reward to centerline once car is within the firsrt linesgement, 
                # e.g. if the agent drives backwards out of map immediately he will always get max reward, because this term will always be 1/12 * self.ditsance_to_center_weight
                return 0 

        absolute_dist = self.env.reference_line.calculate_lateral_difference(idx = next_refline_index, car_position=observations[IPCFields.SIMSTATE].position)
        return self.lateral_distance_manager.scale_lateral_distance(absolute_dist)
    
class TerminationPunishment(BoundedRewardterm):
    """Punihses terminations like 'stuck' or 'no-progress'."""

    def __init__(self, weight):
        super().__init__(weight, "terminations")

    def _get_term(self, observations, processed_obs, race_finished, other_terminations):
        ot = False
        if "stuck" in other_terminations:
            ot = other_terminations["stuck"]
        if "no_progress" in other_terminations:
            ot = ot or other_terminations["no_progress"]

        other_term_reward = ot
        return other_term_reward
    
class HasWallConatact(BoundedRewardterm):

    def __init__(self, weight):
        """Boolean term, returns if 1, if one of the tires has wall contact."""
        super().__init__(weight, "wall_contact")

    def _get_term(self, observations, processed_obs, race_finished, other_terminations):
        #ssD : SimStateData = observations[IPCFields.SIMSTATE]
        #svc : SceneVehicleCar = ssD.scene_mobil 
        return observations[IPCFields.SIMSTATE].scene_mobil.has_any_lateral_contact

class SpeedReward(RewardTerm):
    """Awards display-speed of the player; not weighted, range ~[0,1000]"""

    THEORETICAL_MAX_VALUE = 1000
    """Theroretical maximial Value of the reward-term, before weighting."""
    THEORETICAL_MIN_VALUE = 0
    """Theroretical minimal Value of the reward-term, before weighting."""

    def __init__(self, weight, clip_min=THEORETICAL_MIN_VALUE, clip_max=THEORETICAL_MAX_VALUE):
        super().__init__(weight, "speed", clip_min=clip_min, clip_max=clip_max)

    def _get_term(self, observations, processed_obs, race_finished, other_terminations):
        return observations[IPCFields.SIMSTATE].display_speed 
    
class RaceFinished(BoundedRewardterm):
    """Binary Reward, given if the race is finished."""
    def __init__(self, weight, scaled_by_steps_taken : bool = False):
        """
        Args:
            weight (float) : weight of the reward term used in the sum
            scaled_by_steps_taken (bool) : If True, does not return a boolean, but a fraction 1 - t/x, where t is the amount of steps taken
            and n is the environment-timeout."""
        super().__init__(weight, "race_finished")
        self.scaled_by_steps_taken = scaled_by_steps_taken

    def _get_term(self, observations, processed_obs, race_finished, other_terminations):
        if self.scaled_by_steps_taken:
            # TODO : The maximium reachable reward with this is based on the length of the track (therefore : noramlize relative to track.)
            # (the longer the track, the more the minimal amount of env-steps requrired to reach goal, the less this reward.)
            return (1 - self.env.n_steps / self.env.termination_manager.timeout) * race_finished
        return race_finished
    
class ConstantRewardTerm(BoundedRewardterm):
    """Just returns 1 everystep; i.e. its weighted value is equal to the weight (can be used for consistent time-penalties e.g.)"""
    def __init__(self, weight, name : str = "race_not_finished"):
        super().__init__(weight, name)

    def _get_term(self, observations, processed_obs, race_finished, other_terminations):
        return 1
    

class NoProgressPunishment(RewardTerm):
    """Returns -1, if the agent has not made any progresss along the center-line."""

    THEORETICAL_MAX_VALUE = 0
    """Theroretical maximial Value of the reward-term, before weighting."""
    THEORETICAL_MIN_VALUE = -np.inf
    """Theroretical minimal Value of the reward-term, before weighting."""

    NAME = "no_progress"

    def __init__(self, weight, steps_without_progress_until_punishment = 0, clip_min = -10, clip_max = 0):
        super().__init__(weight, NoProgressPunishment.NAME, clip_min=clip_min, clip_max=clip_max)
        self.current_refline_idx = 0
        self.steps_without_progress_until_punishment = steps_without_progress_until_punishment
        self.steps_since_last_progress = 0

    def _get_term(self, observations, processed_obs, race_finished, other_terminations):
        next_refline_index, _, _ = self.env.reference_line.get_distance_to_next_point()

        if next_refline_index == self.current_refline_idx:
            self.steps_since_last_progress += 1
        else:
            self.steps_since_last_progress = 0

        return (-1) * (self.steps_since_last_progress >= self.steps_without_progress_until_punishment)
    
    def reset(self):
        self.steps_since_last_progress = 0

class OffTrackPunishment(RewardTerm):
    """Implements a punishment if the agent is off-track."""


    THEORETICAL_MAX_VALUE = 0
    """Theroretical maximial Value of the reward-term, before weighting."""
    THEORETICAL_MIN_VALUE = -1
    """Theroretical minimal Value of the reward-term, before weighting."""

    NAME = "off_track"

    def __init__(self, weight, clip_min=THEORETICAL_MIN_VALUE, clip_max=THEORETICAL_MAX_VALUE):
        super().__init__(weight, OffTrackPunishment.NAME, clip_min=clip_min, clip_max=clip_max)
    
    def _get_term(self, observations, processed_obs, race_finished, other_terminations):
        """
        Calculates a penalty if the agent is off-course, based on distance and elevation
        from the reference line. The penalty increases the longer the agent stays off-course.
        """

        sim_state: SimStateData = observations[IPCFields.SIMSTATE]
        current_time = sim_state.time / constants.MILLISECONDS_TO_SECONDS
        delta_time = current_time - self.last_time

        # Get distance and height difference to the reference line
        next_index, lateral_distance, _ = self.env.reference_line.get_distance_to_next_point()
        refline_height = self.env.reference_line.reference_line[next_index][1]
        agent_height = sim_state.position[1]
        height_difference = abs(agent_height - refline_height)

        # Determine if the agent is off-course
        is_off_course = (
            lateral_distance >= constants.MAX_DISTANCE_TO_REFLINE or
            height_difference >= constants.MAX_HEIGHT_DIFERENCE
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

class AccumulatedWallPenalty(RewardTerm):
    THEORETICAL_MAX_VALUE = 0
    """Theroretical maximial Value of the reward-term, before weighting."""
    THEORETICAL_MIN_VALUE = -np.inf
    """Theroretical minimal Value of the reward-term, before weighting."""

    NAME = "accumulated_wall_penalty+"
    def __init__(self, weight, min_wall_contact_time = 0.5 , clip_min : float = -2.0, clip_max = 0.0):
        super().__init__(weight, AccumulatedWallPenalty.NAME, clip_min=clip_min, clip_max=clip_max)

        self.min_wall_contact_time = min_wall_contact_time
    
    def _get_term(self, observations, processed_obs, race_finished, other_terminations):
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