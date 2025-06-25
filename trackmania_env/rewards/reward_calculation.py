from __future__ import annotations
from trackmania_env.utils.position_buffer import PositionBuffer
from tminterface.structs import CheckpointData, SimStateData, CheckpointTime
from game_interaction.ipc_fields import IPCFields
import numpy as np
from numba import jit
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from configs.config import RewardManagerCfg
from trackmania_env.utils.reference_line_manager import ReferenceLineManager

class RewradCalculator:
    """Responsible for reward calculations for environment"""

    def __init__(self, reward_cfg : RewardManagerCfg):
        self.reward_cfg = reward_cfg
        self.pos_buffer = None # do not reset or add anything to this position buffer, read-only! (no reset, no add...)
        self.refline_manager: ReferenceLineManager = None

    def set_position_buffer(self, position_buffer : PositionBuffer):
        """Set position buffer for this instance"""
        self.pos_buffer = position_buffer

    def set_reference_line(self,refline_manager: ReferenceLineManager):
        """ Set reference line for this instance"""
        self.refline_manager = refline_manager

    def calculate_reward(self, observations : dict[str, any], processed_obs : dict[str, any], race_finished : bool, other_terminations : bool) -> tuple[float, dict[str, int | float]]:
        """Calculates the rewrad given observations for current environment-step
        Returns 
        -------
            - reward : which is the cummulative reward of all reward terms
            - reward_info : dictionary containing reward-term-names (str) as keys and the values
                of individual reward terms for this calculation as values (this may also include non-reward values).


        For Future implementations; be sure to only put (str, float/int) pairs into the reward_info-dictionary as this s expected by RewardLogCallback.  
        """
        raise NotImplementedError("Do Not use this class directly. Use RewardCalculator.get_instance()")

    def reset(self) -> None:
        """resets rewrad calculator"""
        pass


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