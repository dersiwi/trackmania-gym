import wandb
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from trackmania_env.utils.return_tracker import ReturnTracker

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
    def __init__(self, verbose=0, n_envs : int = 1):
        super().__init__(verbose)
        self.rewardterms_to_log : list[dict] = [{} for i in range(n_envs)]

    def _on_step(self) -> bool:
        # have to call self.locals["infos"][0], because sb3 has an info-dict for each environment, since currently we only train with one environment, this index is always 0
        #infos : list[dict] = self.locals["infos"][0]
        for env_idx in range(len(self.locals["infos"])):
            env_infos = self.locals["infos"][env_idx]

            if "rewards" in env_infos and not len(env_infos["rewards"]) == 0:

                for rewterm in env_infos["rewards"]:
                    if rewterm in self.rewardterms_to_log[env_idx]:
                        self.rewardterms_to_log[env_idx][rewterm + str(env_idx)] += env_infos["rewards"][rewterm]
                    else:
                        self.rewardterms_to_log[env_idx][rewterm + str(env_idx)] = env_infos["rewards"][rewterm]

            if ("terminated" in env_infos and env_infos["terminated"]) or ("truncated" in env_infos and env_infos["truncated"]):
                wandb.log(self.rewardterms_to_log[env_idx])
                self.rewardterms_to_log[env_idx] = {}

        return True #always return true.


class ReturnCallback(BaseCallback):
    """This callback listens to the environment wirting its episode-return into the infos and then logs this return per episode"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
    
    def _on_step(self):
        infos : list[dict] = self.locals["infos"][0]
        if ReturnTracker.LOG_NAME in infos:
            wandb.log({"episode_return" : infos[ReturnTracker.LOG_NAME]})
        return True # always return true.
    

class ContinuousActionLogCallback(BaseCallback):
    def __init__(self, verbose = 0, log_minmax = True):
        super().__init__(verbose)
        self.actionmin : dict[str, float] = {}
        self.actionmax : dict[str, float] = {}
        self.actionmean : dict[str, float] = {}
        self.minprefix = "min_action_dim:"
        self.maxprefix = "max_action_dim:"
        self.meanprefix = "mean_action_dim:"
        self.n_steps = 0
        self.log_minmax = log_minmax


    def _on_step(self):#info["action"]
        infos : list[dict] = self.locals["infos"][0]
        self.n_steps += 1
        if "action" in infos:
            if not type("action") == np.ndarray:
                pass
            action : np.ndarray = infos["action"]
            for dimidx in range(action.shape[0]):
                actionkey = str(dimidx)
                if actionkey in self.actionmin:
                    self.actionmin[self.minprefix + actionkey] = min(action[dimidx], self.actionmin[self.minprefix + actionkey])
                    self.actionmax[self.maxprefix + actionkey] = max(action[dimidx], self.actionmax[self.maxprefix + actionkey])
                    self.actionmean[self.meanprefix + actionkey] += action[dimidx] / self.n_steps #not reaaally true but close enough
                else:
                    self.actionmin[self.minprefix + actionkey] = action[dimidx]
                    self.actionmax[self.maxprefix + actionkey] = action[dimidx]
                    self.actionmean[self.meanprefix + actionkey] = action[dimidx]
        
        if ("terminated" in infos and infos["terminated"]) or ("truncated" in infos and infos["truncated"]):
            if self.log_minmax:
                wandb.log(self.actionmin)
                wandb.log(self.actionmax)
            wandb.log(self.actionmean)
            self.actionmin : dict[str, float] = {}
            self.actionmax : dict[str, float] = {}
            self.actionmean : dict[str, float] = {}

        return super()._on_step()