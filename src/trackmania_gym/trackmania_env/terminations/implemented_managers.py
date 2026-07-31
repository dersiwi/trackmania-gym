from trackmania_gym.trackmania_env.terminations.termination_manager import TerminationManager
from trackmania_gym.trackmania_env.terminations.termination_terms import NoProgressTerminationTerm

class NoProgressTerminationManager(TerminationManager):
    """This terminatio manager measures the progress of the agent along the centerline, and if there has not been any progress for 
    the last 'terminate_after_steps_without_progress', the enviornment terminates.
    In addition, also if the car is stuck.
    Of course, timeout is calculated."""

    def __init__(self, timeout : int, ignore_stuck_for_n_steps_after_reset:int, terminate_after_steps_without_progress : int):
        super().__init__(timeout, ignore_stuck_for_n_steps_after_reset)
        self.terms = [NoProgressTerminationTerm(terminate_after_steps_without_progress)]