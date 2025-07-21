from __future__ import annotations
from tminterface.structs import CheckpointData, SimStateData, CheckpointTime
import numpy as np
from scipy.stats import norm
import logging
from typing import Literal

from trackmania_env.rewards.reward_calculation import RewradCalculator
from game_interaction.ipc_fields import IPCFields


MAX_LATERAL_DISTANCE = 12 # maximal lateral difference, this is an estimate.

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
                dist_scale: float = 0.3):
        """
        Initializes the reward manager with explicit reward weights and parameters
        for modeling centerline distance reward using a Gaussian function.

        Parameters:
        -----------
            - accum_distance_weight (float):        Weight for accumulated distance reward.
            - race_not_finished_weight (float):     Penalty weight for not finishing the race.
            - race_finished_reward_weight (float):  Bonus reward for finishing the race.
            - other_termination_punishment (float): Penalty for other termination conditions.
            - velocity_reward_weight (float):       Weight for maintaining or increasing velocity.
            - backward_weight (float): Penalty      weight for moving backward.
            - distance_to_center_weight (float):    Weight for staying close to the centerline.
            - velocity_change_reward_weight (float):Weight for smooth velocity changes.
            - speed_reward_weight (float):          General reward for maintaining speed.
            - lateral_distance_mode (str):          Mode used to calculate lateral distance.

            - mean (float)  : Mean of the Gaussian function used for distance-to-centerline reward. (Only active if literal_distance_mode = "Gauss")
                         Represents the ideal centerline offset (typically 0).
            - sigma (float) : Standard deviation of the Gaussian, controlling the reward falloff 
                           as the vehicle deviates from the centerline. (Only active if literal_distance_mode = "Gauss")

            - yshift (float)        : Vertical shift applied to the Gaussian curve to shape the baseline reward.
            - multiplicator (float) : Scales the Gaussian's amplitude; higher values amplify the reward.
            - dist_scale (float)    : Scaling factor for the input distance before applying the Gaussian.
    """
        
        super().__init__()

        self.accum_distance_weight = accum_distance_weight
        self.race_not_finished_weight = race_not_finished_weight
        
        self.race_finished_reward_weight = race_finished_reward_weight
        self.other_termination_punishment = other_termination_punishment
        self.velocity_reward_weight = velocity_reward_weight
        self.backward_weight = backward_weight
        self.distance_to_center_weight = distance_to_center_weight
        self.velocity_change_reward_weight = velocity_change_reward_weight

        self.speed_reward_weight = speed_reward_weight



        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.current_refline_idx : int = 0

        self.max_lateral_difference = MAX_LATERAL_DISTANCE
        self.lateral_distance_mode = lateral_distance_mode
        self.mean, self.sigma, self.yshift, self.multiplicator, self.dist_scale = mean, sigma, yshift, multiplicator, dist_scale


        self.last_simstate : SimStateData = None

    def _get_normed_velocity(self, velocity : list[float, float, float]) -> float:
        return np.linalg.norm(np.array(velocity) / 1000) #1000 == max velocity
    
    def _calculate_accum_distance_reward(self, next_refline_index : int) -> float:
        """Calculates the reward along indicating the distance driven along the centerline"""        
        accum_dist_reward = 0
        if self.current_refline_idx < next_refline_index: #only give reward if progress in regards to last one was made

            for i in range(next_refline_index - self.current_refline_idx):
                accum_dist_reward += self.refline_manager.get_discrete_distance(self.current_refline_idx + i)

            self.current_refline_idx = next_refline_index

        return accum_dist_reward * self.accum_distance_weight
    
    def _calculate_lateral_distance_reward(self, next_refline_idx : int, car_position : np.ndarray) -> float:

        if next_refline_idx <= 0:
            # only calculate reward to centerline once car is within the firsrt linesgement, 
            # e.g. if the agent drives backwards out of map immediately he will always get max reward, because this term will always be 1/12 * self.ditsance_to_center_weight
            return 0 
        
        distance_to_center_reward = 0
        absolute_dist = np.clip(self.refline_manager.calculate_lateral_difference(idx = next_refline_idx, car_position=car_position), a_min = 0, a_max = self.max_lateral_difference)


        if self.lateral_distance_mode == "triangle":
            # inverse the distance, such that reward is bigger once distance gets smaller
            distance_to_center_reward = 0.5 - absolute_dist / self.max_lateral_difference
            
        elif self.lateral_distance_mode == "gauss":
            distance_to_center_reward = self.multiplicator * norm.pdf(absolute_dist * self.dist_scale, loc =self.mean, scale = self.sigma) + self.yshift

        elif self.lateral_distance_mode == "trapez":
            raise NotImplementedError("Not implemented.")

        else:
            raise ValueError(f"No Such Errorfunction '{self.lateral_distance_mode}' available")

        return distance_to_center_reward * self.distance_to_center_weight

    def _calculate_termination_rewards(self, other_terminations) -> float:
        ot = False
        if "stuck" in other_terminations:
            ot = other_terminations["stuck"]
        if "no_progress" in other_terminations:
            ot = ot or other_terminations["no_progress"]

        other_term_reward = (-1) * ot * self.other_termination_punishment
        return other_term_reward
        
    def calculate_reward(self, observations, processed_obs : dict[str, any], race_finished, other_terminations : dict[str, bool]):
        ssD : SimStateData = observations[IPCFields.SIMSTATE]
        reward = 0
        
        next_refline_index, d, drel = self.refline_manager.get_distance_to_next_point()
        accum_dist_reward = self._calculate_accum_distance_reward(next_refline_index)

        #current_velocity_normed = self._get_normed_velocity(ssD.velocity)
        #velocity_change_reward = 0
        #if not self.last_simstate == None:
        #    velocity_change_reward = (current_velocity_normed - self._get_normed_velocity(self.last_simstate.velocity)) * self.velocity_change_reward_weight

        speed_reward = ssD.display_speed * self.speed_reward_weight / 1000 #TODO figure out good weight

        distance_to_center_reward = self._calculate_lateral_distance_reward(next_refline_index, ssD.position)
            
        #velocity_reward = self.velocity_reward_weight * current_velocity_normed
        race_not_finished_reward = (-1) * self.race_not_finished_weight
        race_finished = race_finished * self.race_finished_reward_weight
        other_term_reward = self._calculate_termination_rewards(other_terminations)
        #backward_punishment = (-1) * np.clip(d, a_min=0, a_max=100) / 100 * self.backward_weight
        
        reward = accum_dist_reward + race_not_finished_reward + race_finished + other_term_reward + distance_to_center_reward + speed_reward 
        self.last_simstate = ssD
        return reward, {"total" : reward, 
                        "accumulated_distance" : accum_dist_reward,
                        "distance_to_center" : distance_to_center_reward,
                        "nextpoint_reference_index" : next_refline_index,
                        "race_not_finished":race_not_finished_reward,
                        "race_finished" : race_finished,
                        "other_terminations":other_term_reward,
                        "speed_reward":speed_reward}




    def reset(self):
        self.current_refline_idx = 0
        return super().reset()
    

class NextPointRewards2(NextPointRewards):
    """Similar to NextPointRewards, but it scales the distance to center in relation to the accumulated distance."""
    def __init__(self, sf1: float = 0.4,sf2: float = 1.0,**kwargs):
        """
            sf1 (float): Scale factor for how large the distance-to-center reward can be
                         relative to the accumulated distance reward (default: 0.4).
            sf2 (float): Additional scaling for distance-to-center reward (default: 1.0).
        """
        super().__init__(**kwargs)
        self.sf1 = 0.4
        """Scale factor of how large distance-to-center reward can be in relation to accum distance reward.""" 
        self.sf2 = 1.0
        """Scale factor of how large distance-to-center reward can be in relation to accum distance reward.""" 

    def calculate_reward(self, observations, processed_obs, race_finished, other_terminations):
        ssD : SimStateData = observations[IPCFields.SIMSTATE]
        reward = 0
        
        next_refline_index, d, drel = self.refline_manager.get_distance_to_next_point()
        accum_dist_reward = self._calculate_accum_distance_reward(next_refline_index)
        distance_to_center_reward = self._calculate_lateral_distance_reward(next_refline_index, ssD.position)
        if not distance_to_center_reward == 0:
            distance_to_center_reward_sign = distance_to_center_reward / abs(distance_to_center_reward)
            distance_to_center_reward_abs = min(abs(distance_to_center_reward), self.sf1 * accum_dist_reward)
            distance_to_center_reward = distance_to_center_reward_abs * distance_to_center_reward_sign
        
        other_term_reward = self._calculate_termination_rewards(other_terminations)

        speed_reward = ssD.display_speed * self.speed_reward_weight / 1000
        speed_reward = min(speed_reward, self.sf2 * accum_dist_reward)
        

        reward = speed_reward + accum_dist_reward + distance_to_center_reward + other_term_reward

        return reward, {"speed_reward" : speed_reward,
                        "accum_dist_reward" : accum_dist_reward,
                        "distance_to_center_reward" : distance_to_center_reward,
                        "other_terminations" : other_term_reward}
    


class RaceFinishedRewards(NextPointRewards):
    """Similar to NextPointRewards, but it scales the distance to center in relation to the accumulated distance."""
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.env_timeout = 0
    
        """After how many (env)-steps the environment times out"""

    def set_env(self, env):
        super().set_env(env)
        self.env_timeout = self.env.termination_manager.timeout


    def calculate_reward(self, observations, processed_obs, race_finished, other_terminations):
        reward = 0
        
        next_refline_index, d, drel = self.refline_manager.get_distance_to_next_point()
        accum_dist_reward = self._calculate_accum_distance_reward(next_refline_index)
        
        other_term_reward = self._calculate_termination_rewards(other_terminations)
        race_finished_reward = 0

        if race_finished:
            # TODO : The maximium reachable reward with this is based on the length of the track (therefore : noramlize relative to track.)
            # (the longer the track, the more the minimal amount of env-steps requrired to reach goal, the less this reward.)
            race_finished_reward = (1 - self.env.n_steps / self.env_timeout) * self.race_finished_reward_weight

        reward = accum_dist_reward + other_term_reward + race_finished_reward

        return reward, {"accum_dist_reward" : accum_dist_reward,
                        "race_finished_reward" : race_finished_reward,
                        "other_terminations" : other_term_reward}