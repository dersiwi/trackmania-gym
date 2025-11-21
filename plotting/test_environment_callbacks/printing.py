from plotting.test_environment_callbacks.core import TestEnvironmentCallback

class PrintRewardsToConsole(TestEnvironmentCallback):

    def __init__(self):
        super().__init__()
        self.accumulated = 0

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        if self.n_step % 128 == 0:
            for key in info["rewards"]:
                print(key, end=" | ")
            print("\n")
        for key in info["rewards"]:
            print(key,info["rewards"][key], end=" | ")
            if "total" in info["rewards"]:
                self.accumulated += info["rewards"]["total"]
                print(f"Accumulated : {self.accumulated}")
        print("\n")
        self.n_step += 1

    def reset(self):
        self.accumulated = 0