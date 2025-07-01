from trackmania_env.terminations.termination_manager import TerminationManager


class TimeoutTerminationManager(TerminationManager):
    """This termination manager increases the timeouts after given intervals (aka. steps by the environment)"""

    def __init__(self,timeout: int, ignore_stuck_for_n_steps_after_reset: int, increase_timeout_intervals : list[int], new_timeouts : list[int]):
        super().__init__(timeout, ignore_stuck_for_n_steps_after_reset)

        self.increase_timeout_intervals = increase_timeout_intervals
        self.new_timeouts = new_timeouts
        self.increase_idx = 0
        

    def update_timeout(self) -> int:
        """Updates the current timeout for the environment. """

        if self.increase_idx < len(self.increase_timeout_intervals) and self.env.total_steps >= self.increase_timeout_intervals[self.increase_idx]:
            self.timeout = self.new_timeouts[self.increase_idx]
            self.increase_idx += 1    

    def calculate_terminations(self, observations):
        self.update_timeout()
        truncated = super().calculate_timeout()
        terminated = super().car_is_stuck()
        return terminated, truncated, {"stuck" : terminated, "timeout" : truncated}