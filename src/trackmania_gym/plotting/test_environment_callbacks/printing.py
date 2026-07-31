from trackmania_gym.plotting.test_environment_callbacks.core import TestEnvironmentCallback
from trackmania_gym.trackmania_env.envs.info import EnvironmentInfo
class PrintRewardsToConsole(TestEnvironmentCallback):

    def __init__(self):
        super().__init__()
        self.accumulated = 0

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        if self.n_step % 128 == 0:
            for key in info[EnvironmentInfo.REWARDS]:
                print(key, end=" | ")
            print("\n")
        for key in info[EnvironmentInfo.REWARDS]:
            print(key,info[EnvironmentInfo.REWARDS][key], end=" | ")
            if "total" in info[EnvironmentInfo.REWARDS]:
                self.accumulated += info[EnvironmentInfo.REWARDS]["total"]
                print(f"Accumulated : {self.accumulated}")
        print("\n")
        self.n_step += 1

    def reset(self):
        self.accumulated = 0