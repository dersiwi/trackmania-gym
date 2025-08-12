
from stable_baselines3.common.running_mean_std import RunningMeanStd
import numpy as np



class RewardNormalizer:
    """Normalizes rewards using a running average"""

    def __init__(self, eps : float = 0.0001):
        """
        Agrs:
            eps (float): For numerical stability (prevent div by zero)"""
        self.rms = RunningMeanStd(epsilon = eps)
        self.epsilon = eps

    def normalize_float(self, rewards : float) -> np.ndarray:
        """Noramlize the rewards using running average and variance: 
            (r - mean) / (sqrt(var) + eps) 
        Update Running average and variance after normalizing.
        
        Args:
            rewards (float) : Unnormalized rewards
        Returns
            normalized rewards (float)
        """
        rewards = np.array([rewards])
        normalized = (rewards - self.rms.mean) / (np.sqrt(self.rms.var) + self.epsilon)
        self.rms.update(rewards)
        return normalized
    

if __name__ == "__main__":
    rn = RewardNormalizer()#
    for i in range(200):
        rew = 26
        print(rn.normalize_float(rew))