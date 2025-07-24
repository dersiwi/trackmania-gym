
from stable_baselines3.common.running_mean_std import RunningMeanStd
import numpy as np



class RewardNormalizer:
    """Normalizes rewards using a runnin average"""

    def __init__(self, eps : float = 0.0001):
        """eps : For numerical stability."""
        self.rms = RunningMeanStd(epsilon = eps)
        self.epsilon = eps

    def normalize_float(self, rewards : float) -> np.ndarray:
        rewards = np.array([rewards])
        normalized = (rewards - self.rms.mean) / (np.sqrt(self.rms.var) + self.epsilon)
        self.rms.update(rewards)
        return normalized
    

if __name__ == "__main__":
    rn = RewardNormalizer()#
    for i in range(200):
        rew = 26
        print(rn.normalize_float(rew))