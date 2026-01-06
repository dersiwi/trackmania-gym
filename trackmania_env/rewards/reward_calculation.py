from __future__ import annotations
import wandb
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

from trackmania_env.utils.reference_line_manager import ReferenceLineManager
from trackmania_env.utils.position_buffer import PositionBuffer
from trackmania_env.rewards.normalizer import RewardNormalizer
from trackmania_env.manager import Manager, ManagerTerm

class RewardTerm(ManagerTerm):

    THEORETICAL_MAX_VALUE = np.inf
    """Theroretical maximial Value of the reward-term, before weighting."""
    THEORETICAL_MIN_VALUE = -np.inf
    """Theroretical minimal Value of the reward-term, before weighting."""


    def __init__(self, weight : float, name : str, clip_min : float, clip_max : float):
        """
        Args:
            weight (float)  : factor by which the reward is multiplied before being added to other reward terms to form a final reward signal
            name (str)      : Name of the reward-term
            clip_min (flaot): Minimal value of the reward term (applied before weighing)
            clip_max (float): Maximal value of the reward term (applied before weighing)
        """
        super().__init__(name)
        self.weight = weight
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.env = None

        
    def _get_term(self, observations : dict[str, any], processed_obs : dict[str, any], race_finished : bool, other_terminations : dict[str, bool]) -> float:
        raise NotImplementedError("")

    def _get_weighted_term(self, observations : dict[str, any], processed_obs : dict[str, any], race_finished : bool, other_terminations : dict[str, bool]) -> float:
        termval = self._get_term(observations, processed_obs, race_finished, other_terminations)
        clipped_val = np.clip(termval, a_min = self.clip_min, a_max=self.clip_max)
        return clipped_val * self.weight
    
    def calculate_reward_term(self, observations : dict[str, any], processed_obs : dict[str, any], race_finished : bool, other_terminations : dict[str, bool]) -> tuple[float, dict[str, int | float]]:
        val = self._get_weighted_term(observations, processed_obs, race_finished, other_terminations)
        return val, {self.name : val}


class BoundedRewardterm(RewardTerm):
    """The term itself can only return values 0,1."""

    THEORETICAL_MAX_VALUE = 1
    """Theroretical maximial Value of the reward-term, before weighting."""
    THEORETICAL_MIN_VALUE = 0
    """Theroretical minimal Value of the reward-term, before weighting."""

    def __init__(self, weight, name):
        super().__init__(weight, name, clip_min=0.0, clip_max=1.0)

class RewradCalculator(Manager):
    """Responsible for reward calculations for environment.
    Subclasses have to initialize self.terms with the desired reward-terms. This has to happen in the constructor!"""

    def __init__(self, normalize : bool = False):
        self.env = None
        self.pos_buffer = None # do not reset or add anything to this position buffer, read-only! (no reset, no add...)
        self.refline_manager: ReferenceLineManager = None
        self.normalizer = RewardNormalizer()
        self.normalize = normalize

        self.terms : list[RewardTerm] #this is just for type-inference.

    def get_sum_of_weighted_rewards(self, observations : dict[str, any], processed_obs : dict[str, any], race_finished : bool, other_terminations : dict[str, bool]) -> tuple[float, dict[str, int | float]]:
        rew = 0
        info = {}
        for term in self.terms:
            try:
                termvalue, terminfo = term.calculate_reward_term(observations, processed_obs, race_finished, other_terminations)
            except Exception as e:
                self.logger.error(f"Failure to retrieve reward-term {term.name}, error-traceback : {e} \n\n Supplementing reward-term with zeros.")
                termvalue, terminfo = 0, {term.name : 0}
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
            reward (float) : Sum of weighted rewards (uses get_sum_of_weighted_rewards).
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