from trackmania_env.terminations.termination_manager import TerminationManager
from trackmania_env.terminations.timeout_policy import TimeoutTerminationManager
from trackmania_env.terminations.noprogress_termination import NoProgressTerminationManager
from configs.config import TrainConfig

def get_termination_manager(termination_cfg : TrainConfig) -> TerminationManager:
    """Returns instance of termination manager according to configuraion"""
    name = termination_cfg.name
    if name == "timeout_increase":
        return TimeoutTerminationManager(timeout = termination_cfg.max_steps_until_reset, 
                                         ignore_stuck_for_n_steps_after_reset=termination_cfg.ignore_stuck_for_n_steps_after_reset, 
                                         increase_timeout_intervals=termination_cfg.increase_timeout_intervals, 
                                         new_timeouts=termination_cfg.new_timeouts)
    elif name == "no_progress":
        return NoProgressTerminationManager(timeout = termination_cfg.max_steps_until_reset, 
                                         ignore_stuck_for_n_steps_after_reset=termination_cfg.ignore_stuck_for_n_steps_after_reset, 
                                         terminate_after_steps_without_progress=termination_cfg.terminate_after_steps_without_progress)
    
    else:
        raise ValueError(f"Termination-Manager '{name}' not known.")