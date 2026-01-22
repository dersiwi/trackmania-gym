import numpy as np
import wandb

from stable_baselines3.common.callbacks import BaseCallback


class ReturnTracker:
    """Calcualtes returns by receiving rewards from a RL environment every timestep. Discounts them accordingly."""
    
    LOG_NAME = "episode-return"
    """name for the return in the info-dict returned by environment. May only exist at the end of an episode."""
    
    def __init__(self, length : int, gamma : float):
        """
        Args:
            length (int)    : how many timesteps back this tracker logs the returns
            gamma (float)   : Discount factor by which rewards is multiplied.
        """
        self.discounted_rewards = np.zeros(length)
        self.gamma = gamma

    def add_reward(self, reward : float) -> None:
        """Adds a new reward to the return
        Args:
            reward (float)  : Reward of the current environment step """
        self.discounted_rewards[1:] = self.discounted_rewards[:-1] * self.gamma
        self.discounted_rewards[0] = reward.item()


    def get_return(self) -> np.ndarray:
        """Returns the calcualted return
        Returns:
            Reinfocement-Learning-Return (float) : sum(r_0, ..., gamma^length * r_length), where length is a class variable
        """
        return np.sum(self.discounted_rewards)
    
    
    def reset(self) -> None:
        """Resets the return-tracker."""
        self.discounted_rewards = np.zeros_like(self.discounted_rewards)
    
class ReturnLogCallback(BaseCallback):
    """
    This custom ReturnLogCallback logs the discounted returns of the agent at the end of each episode.
    """
    def __init__(self, verbose=0):
        return super().__init__(verbose)

    def _on_step(self) -> bool:
        # have to call self.locals["infos"][0], because sb3 has an info-dict for each environment, since currently we only train with one environment, this index is always 0
        infos : list[dict] = self.locals["infos"][0]

        if ReturnTracker.LOG_NAME in infos:
            wandb.log({ReturnTracker.LOG_NAME : infos[ReturnTracker.LOG_NAME]})

        return True #always return true.



if __name__ == "__main__":
    rt = ReturnTracker(5, 0.5)
    for i in range(5):
        rt.add_reward(1)
        print(rt.discounted_rewards)
        print(rt.get_return())

    rt.reset()

    for i in range(5):
        rt.add_reward(1)
        print(rt.discounted_rewards)
        print(rt.get_return())