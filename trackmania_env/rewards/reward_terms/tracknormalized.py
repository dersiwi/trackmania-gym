
from trackmania_env.rewards.reward_calculation import RewardTerm, BoundedRewardterm
from trackmania_env.utils import constants
from trackmania_env.utils.lateral_distance_manager import LateralDistanceManager
from game_interaction.ipc_fields import IPCFields
import numpy as np


class AccumulatedTotal(RewardTerm):
    """Calculates the reward along indicating the distance driven along the centerline"""

    NAME = "accumulated_total"

    def __init__(self, weight, total_reward : float, enhanced_by_amount_travelled : bool, exponential_factor : float):
        """Iniitialites
        Args:
            weight (float)  : Weight of the term
            enhanced_by_amount_travelled (bool) : If true, multiplies the reward by the amount of refline-points passed (to give more incentive to pass more at a single environment step)
            exponential_factor (float)          : Number of steps travelled in this step is raised to this power. Default 1.
        """
        super().__init__(weight, AccumulatedTotal.NAME, clip_min=0, clip_max=1.0)
        self.current_refline_idx = 0
        self.total_reward = total_reward
        self.enhanced_by_amount_travelled = enhanced_by_amount_travelled
        self.exponential_factor=exponential_factor
        self.last_pos = None
        self.rew_per_point = 0

    def set_env(self, env):
        from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
        env : TMNF_Single_Agent_Env = env
        self.rew_per_point = self.total_reward / env.reference_line.n_reference_points
        return super().set_env(env)

    def _get_term(self, observations : dict[str, any], processed_obs : dict[str, any], race_finished : bool, other_terminations : dict[str, bool]):
        next_refline_index, _, _ = self.env.reference_line.get_distance_to_next_point()
        accum_dist_reward = 0
        n_passed = next_refline_index - self.current_refline_idx
        if self.current_refline_idx < next_refline_index: #only give reward if progress in regards to last one was made
            accum_dist_reward = n_passed * self.rew_per_point                
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