from tminterface.structs import CheckpointData, SimStateData, CheckpointTime
from trackmania_env.terminations.termination_terms import TerminationTerm
from trackmania_env.manager import Manager

class TerminationManager(Manager):

    def __init__(self, timeout : int, ignore_stuck_for_n_steps_after_reset : int):

        self.timeout : int = timeout
        self.ignore_stuck_for_n_steps_after_reset = ignore_stuck_for_n_steps_after_reset

        self.terms : list[TerminationTerm] = [] #just for typeinference.

    def calculate_terminations(self, observations : SimStateData) -> tuple[bool, bool, dict[str, bool]]:
        """Returns terminated, truncated for environment step."""
        
        stuck = self.car_is_stuck()
        timeout = self.calculate_timeout()
        terminated, truncated = stuck, timeout
        termination_dict = {"stuck" : stuck, "timeout" : timeout}
        for term in self.terms:
            term_terminated, term_truncated = term.calculate_termination(observations)
            termination_dict.update({term.name : term_terminated or term_truncated})
            terminated = terminated or term_terminated
            truncated = truncated or term_truncated

        return terminated, truncated, termination_dict

    # Implementation of default-termination terms (i.e. stuck and timeout.)

    def calculate_timeout(self) -> bool:
        """Calculates if timeout has been reached"""
        return self.env.n_steps >= self.timeout
    

    def car_is_stuck(self) -> bool:
        stuck = False if self.env.n_steps < self.ignore_stuck_for_n_steps_after_reset else not self.env.position_buffer.moved_more_than_threshold(self.env.position_buffer_threshold)
        return stuck



