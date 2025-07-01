from trackmania_env.terminations.termination_manager import TerminationManager


class NoProgressTerminationManager(TerminationManager):
    """This terminatio manager measures the progress of the agent along the centerline, and if there has not been any progress for 
    the last 'terminate_after_steps_without_progress', the enviornment terminates.
    In addition, also if the car is stuck.
    Of course, timeout is calculated."""

    def __init__(self, timeout : int, ignore_stuck_for_n_steps_after_reset:int, terminate_after_steps_without_progress : int):
        super().__init__(timeout, ignore_stuck_for_n_steps_after_reset)

        self.idx_since_last_advance = 0
        self.n_steps_since_last_progress = 0
        self.terminate_after_steps_without_progress = terminate_after_steps_without_progress

    def calculate_progress(self) -> bool:
        i, _, _ = self.env.reference_line.get_distance_to_next_point()
        # figure out if car has made any progress
        if i > self.idx_since_last_advance:
            self.n_steps_since_last_progress = 0
            self.idx_since_last_advance = i
        else:
            self.n_steps_since_last_progress += 1
        no_progress = self.n_steps_since_last_progress >= self.terminate_after_steps_without_progress
        return no_progress

    def calculate_terminations(self, observations):
        progress = self.calculate_progress()
        stuck = super().car_is_stuck()
        timeout = super().calculate_timeout()
        return progress or stuck, timeout,  {"stuck" : stuck, "no_progress" : progress, "timeout" : timeout}

    def reset(self):
        self.idx_since_last_advance = 0
        self.n_steps_since_last_progress = 0