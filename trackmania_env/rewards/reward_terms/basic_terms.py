"""
Contains basic reward terms
    - AccumulatedDistanceReward
    - LateralDistanceReward
    - TerminationPunishment
    - SpeedReward
    - RaceFinished
    - ConstantRewardTerm
    - NoProgressPunishment
    - OffTrackPunishment
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
        super().__init__(weight, AccumulatedDistanceReward.NAME, clip_min=0, clip_max=1.0)
        self.current_refline_idx = 0
        self.enhanced_by_amount_travelled = enhanced_by_amount_travelled
        self.exponential_factor = exponential_factor
        self.last_pos = None

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

        accum_dist_reward = self._check_against_lag(accum_dist_reward, observations, n_passed)
        return accum_dist_reward
    
    def _check_against_lag(self, accum_dist_reward : float, observations : dict[str, any], n_passed : int) -> float:
        """
        This method checks for lag and if there was some, sets the calcualted accum_distance reward to 0. When training a policy, it is policy that the policy-update 
        takes longer than the process-wrapper can possible wait on another action; in this case the pw sends no-action command to the plugin, which reults in the car
        doing nothing anymore; but this (if the car was driving forward, pushes the car forward.) -> this would trigger a reward over many more refline points within
        one env-step. 
        This method prevents it.
        """

        currpos = np.array(observations[IPCFields.SIMSTATE].position)
        #if not self.last_pos is None: print(currpos, self.last_pos, np.linalg.norm(currpos - self.last_pos), accum_dist_reward, n_passed)
        if not self.last_pos is None and np.linalg.norm(currpos - self.last_pos) > 5 or n_passed > 20:
            """
            - np.linalg.norm(currpos - self.last_pos) > 5 : is an estimated threshold of how fast the agent can go within one ENV-Step 
            - n > 20 : Basically the same as above, but interms of reference line points (this is, if the lookahead size of reflinepoint manager is too small, then this condition
                still catches the lag.)
                
             TODO : This IS dependent on the actions-per-second, however should only fail (it at all) if the APS decrease"""
            #print("Setting to zero")
            accum_dist_reward = 0
        self.last_pos = currpos
        return accum_dist_reward

    
    def reset(self):
        self.current_refline_idx = 0
        self.last_pos = None
    
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

class DriveForwardReward(RewardTerm):
    def __init__(self, weight):
        super().__init__(weight, "forward_rew", 0.0, 1.0)

    def set_env(self, env):
        assert not env.is_discrete, f"This reward term requires a continuous action-space, but got env.is_discrete={env.is_discrete}"
        return super().set_env(env)

    def _get_term(self, observations, processed_obs, race_finished, other_terminations):
        if len(self.env.actions) == 0:
            return 0.0
        return 1.0 if self.env.actions[-1][1] > 0 else 0.0
    
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