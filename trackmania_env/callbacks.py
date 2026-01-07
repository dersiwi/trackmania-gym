import wandb
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


class ReturnCallback(BaseCallback):
    """This callback listens to the environment wirting its episode-return into the infos and then logs this return per episode"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
    
    def _on_step(self):
        infos : list[dict] = self.locals["infos"][0]
        if ReturnTracker.LOG_NAME in infos:
            wandb.log(infos[ReturnTracker.LOG_NAME])
        return True # always return true.