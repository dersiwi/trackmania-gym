from __future__ import annotations
import wandb
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

from trackmania_env.utils.reference_line_manager import ReferenceLineManager
from trackmania_env.utils.position_buffer import PositionBuffer
from trackmania_env.rewards.normalizer import RewardNormalizer

class RewardTerm():

    def __init__(self, weight : float, name : str):
        self.weight = weight
        self.name = name
        self.env = None

    def _get_term(self, observations : dict[str, any], processed_obs : dict[str, any], race_finished : bool, other_terminations : dict[str, bool]) -> float:
        raise NotImplementedError("")

    def _get_weighted_term(self, observations : dict[str, any], processed_obs : dict[str, any], race_finished : bool, other_terminations : dict[str, bool]) -> float:
        return self._get_term(observations, processed_obs, race_finished, other_terminations) * self.weight
    
    def calculate_reward_term(self, observations : dict[str, any], processed_obs : dict[str, any], race_finished : bool, other_terminations : dict[str, bool]) -> tuple[float, dict[str, int | float]]:
        val =  self._get_weighted_term(observations, processed_obs, race_finished, other_terminations)
        return val, {self.name : val}
    
    def set_env(self, env):
        from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
        self.env : TMNF_Single_Agent_Env= env
        
    def reset(self) -> None:
        pass

class RewradCalculator:
    """Responsible for reward calculations for environment.
    Subclasses have to implement self.get_sum_of_weighted_rewards()"""

    def __init__(self, normalize : bool = False):
        self.env = None
        self.pos_buffer = None # do not reset or add anything to this position buffer, read-only! (no reset, no add...)
        self.refline_manager: ReferenceLineManager = None
        self.normalizer = RewardNormalizer()
        self.normalize = normalize

        self.reward_terms : list[RewardTerm] = []


    def set_env(self, env):
        from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
        self.env : TMNF_Single_Agent_Env= env

        for term in self.reward_terms:
            term.set_env(env)

    def get_sum_of_weighted_rewards(self, observations : dict[str, any], processed_obs : dict[str, any], race_finished : bool, other_terminations : dict[str, bool]) -> tuple[float, dict[str, int | float]]:
        rew = 0
        info = {}
        for term in self.reward_terms:
            termvalue, terminfo = term.calculate_reward_term(observations, processed_obs, race_finished, other_terminations)
            rew += termvalue
            info = info | terminfo
        info["total"] = rew
        return rew, info
    
    def calculate_reward(self, observations : dict[str, any], processed_obs : dict[str, any], race_finished : bool, other_terminations : dict[str, bool]) -> tuple[float, dict[str, int | float]]:
        """Calculates the rewrad given observations for current environment-step
        
        Args:
            observations  (dict)    : raw observations obtained from game
            provessed_obs (dict)    : observations after being processed by manager
            race_finished (bool)    : True, if race has been finished organically
            other_terminations (dict): Information dict given by Termination Manager; contains information about other possible terminatio reasons.

        Returns:
            reward (float) : Sum of weighted rewards; 
            - self.normalize_reward(reward) : which is the cummulative reward of all reward terms, normalized, depending on the 
                initialization of this class.
            - reward_info : dictionary containing reward-term-names (str) as keys and the values
                of individual reward terms for this calculation as values (this may also include non-reward values).


        For Future implementations; be sure to only put (str, float/int) pairs into the reward_info-dictionary as this s expected by RewardLogCallback.  
        """
        weighted_rewards, info = self.get_sum_of_weighted_rewards(observations, processed_obs, race_finished, other_terminations)
        return self.normalize_reward(weighted_rewards), info
    
    def normalize_reward(self, rewards : float) -> float | np.ndarray:
        """Normalizes rewards, if self.normalize == True. If not, just returns them as is"""
        if self.normalize:
            return self.normalizer.normalize_float(rewards)
        return rewards

    def reset(self) -> None:
        """resets rewrad calculator"""
        for term in self.reward_terms:
            term.reset()


class RewardLogCallback(BaseCallback):
    """
    This custom RewardLogCallback should log the rewards on a per-step basis and also log each reward-term individually.
    """
    def __init__(self, verbose=0):
        return super().__init__(verbose)

    def _on_step(self) -> bool:
        # have to call self.locals["infos"][0], because sb3 has an info-dict for each environment, since currently we only train with one environment, this index is always 0
        infos : list[dict] = self.locals["infos"][0]

        if "rewards" in infos and not len(infos["rewards"]) == 0:
            wandb.log(infos["rewards"])

        return True #always return true.
    

class AccumRewardLogCallback(BaseCallback):
    """
    This custom RewardLogCallback should log the individual, accumulated reward-terms after each episode ends.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewardterms_to_log = {}

    def _on_step(self) -> bool:
        # have to call self.locals["infos"][0], because sb3 has an info-dict for each environment, since currently we only train with one environment, this index is always 0
        infos : list[dict] = self.locals["infos"][0]

        if "rewards" in infos and not len(infos["rewards"]) == 0:

            for rewterm in infos["rewards"]:
                if rewterm in self.rewardterms_to_log:
                    self.rewardterms_to_log[rewterm] += infos["rewards"][rewterm]
                else:
                    self.rewardterms_to_log[rewterm] = infos["rewards"][rewterm]

        if ("terminated" in infos and infos["terminated"]) or ("truncated" in infos and infos["truncated"]):
            wandb.log(self.rewardterms_to_log)
            self.rewardterms_to_log = {}

        return True #always return true.