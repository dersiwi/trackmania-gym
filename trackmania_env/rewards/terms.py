from trackmania_env.rewards.reward_calculation import RewardTerm
from trackmania_env.utils.lateral_distance_manager import LateralDistanceManager
from game_interaction.ipc_fields import IPCFields

class AccumulatedDistanceReward(RewardTerm):
    """Calculates the reward along indicating the distance driven along the centerline"""

    def __init__(self, weight):
        super().__init__(weight, "accumulated_distance")
        self.current_refline_idx = 0

    def _get_term(self, observations : dict[str, any], processed_obs : dict[str, any], race_finished : bool, other_terminations : dict[str, bool]):
        next_refline_index, _, _ = self.env.reference_line.get_distance_to_next_point()
        accum_dist_reward = 0
        if self.current_refline_idx < next_refline_index: #only give reward if progress in regards to last one was made

            for i in range(next_refline_index - self.current_refline_idx):
                accum_dist_reward += self.env.reference_line.get_discrete_distance(self.current_refline_idx + i)

            self.current_refline_idx = next_refline_index

        return accum_dist_reward
    
    def reset(self):
        self.current_refline_idx = 0
    

class LateralDistanceReward(RewardTerm):

    def __init__(self, weight, lateral_distance_mode : str):
        super().__init__(weight, "lateral_distance")
        self.lateral_distance_manager = LateralDistanceManager.get_instance(lateral_distance_mode)

    def _get_term(self, observations : dict[str, any], processed_obs : dict[str, any], race_finished : bool, other_terminations : dict[str, bool]):
        next_refline_index, _, _ = self.env.reference_line.get_distance_to_next_point()
        if next_refline_index <= 0:
                # only calculate reward to centerline once car is within the firsrt linesgement, 
                # e.g. if the agent drives backwards out of map immediately he will always get max reward, because this term will always be 1/12 * self.ditsance_to_center_weight
                return 0 

        absolute_dist = self.env.reference_line.calculate_lateral_difference(idx = next_refline_index, car_position=observations[IPCFields.SIMSTATE].position)
        return self.lateral_distance_manager.scale_lateral_distance(absolute_dist)
    

class TerminationPunishment(RewardTerm):

    def __init__(self, weight):
        super().__init__(weight, "terminations")

    def _get_term(self, observations, processed_obs, race_finished, other_terminations):
        ot = False
        if "stuck" in other_terminations:
            ot = other_terminations["stuck"]
        if "no_progress" in other_terminations:
            ot = ot or other_terminations["no_progress"]

        other_term_reward = (-1) * ot
        return other_term_reward

class SpeedReward(RewardTerm):
    def __init__(self, weight):
        super().__init__(weight, "speed")

    def _get_term(self, observations, processed_obs, race_finished, other_terminations):
        return observations[IPCFields.SIMSTATE].display_speed 
    
class RaceFinished(RewardTerm):
    def __init__(self, weight, scaled_by_steps_taken : bool = False):
        super().__init__(weight, "race_finished")
        self.scaled_by_steps_taken = scaled_by_steps_taken

    def _get_term(self, observations, processed_obs, race_finished, other_terminations):
        if self.scaled_by_steps_taken:
            # TODO : The maximium reachable reward with this is based on the length of the track (therefore : noramlize relative to track.)
            # (the longer the track, the more the minimal amount of env-steps requrired to reach goal, the less this reward.)
            return (1 - self.env.n_steps / self.env.termination_manager.timeout) * race_finished
        return race_finished
    
class ConstantRewardTerm(RewardTerm):
    def __init__(self, weight):
        super().__init__(weight, "race_not_finished")

    def _get_term(self, observations, processed_obs, race_finished, other_terminations):
        return 1
    

class NoProgressPunishment(RewardTerm):

    def __init__(self, weight, steps_without_progress_until_punishment = 0):
        super().__init__(weight, "no_progress")
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