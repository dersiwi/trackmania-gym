

class TimeoutPolicy:

    def __init__(self, increase_timeout_intervals : list[int], new_timeouts : list[int]):
        self.total_env_timesteps = 0
        self.increase_timeout_intervals = increase_timeout_intervals
        self.new_timeouts = new_timeouts
        self.increase_idx = 0
        

    def update_timeout(self, current_timeout : int) -> int:
        """Updates the current timeout for the environment.
         - current_timeout : current timeout of the environment; aka after how many steps "trucated" is true
         
          returns : new timeout for environment according to internal logic. """
        self.total_env_timesteps += 1

        if self.increase_idx >= len(self.increase_timeout_intervals):
            return current_timeout

        if self.total_env_timesteps >= self.increase_timeout_intervals[self.increase_idx]:
            #print(f"-----------------INCREASING TIMEOUT FROM {current_timeout} TO {self.new_timeouts[self.increase_idx]} after {self.total_env_timesteps} env-steps !!!")
            current_timeout = self.new_timeouts[self.increase_idx]
            self.increase_idx += 1
        return current_timeout