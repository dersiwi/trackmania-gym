from abc import ABC, abstractmethod
from trackmania_gym.trackmania_env.manager import ManagerTerm

class TerminationTerm(ABC, ManagerTerm):

    def __init__(self, name):
        super().__init__(name)

    @abstractmethod
    def calculate_termination(self, observations) -> tuple[bool, bool]:
        """Calculates if the termination term is a terminated, or truncated and returns that signal.
        Args:
            observations (processed-observations) : This is the observation-signal after being processed by the observation-manager
        Returns:
            tuple (tuple[bool]) : Tuple containing terminated and truncated for that termination term"""
        raise NotImplementedError()


class NoProgressTerminationTerm(TerminationTerm):
    def __init__(self, terminate_after_steps_without_progress : int):
        super().__init__("no_progress")
        self.terminate_after_steps_without_progress = terminate_after_steps_without_progress
        self.idx_since_last_advance = 0
        self.n_steps_since_last_progress = 0

    def calculate_termination(self, observations):
        i, _, _ = self.env.reference_line.get_distance_to_next_point()
        # figure out if car has made any progress
        if i > self.idx_since_last_advance:
            self.n_steps_since_last_progress = 0
            self.idx_since_last_advance = i
        else:
            self.n_steps_since_last_progress += 1
        no_progress = self.n_steps_since_last_progress >= self.terminate_after_steps_without_progress
        return no_progress, False   #is never truncated.
    
    def reset(self):
        self.idx_since_last_advance = 0