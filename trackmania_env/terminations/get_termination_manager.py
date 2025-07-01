from trackmania_env.terminations.termination_manager import TerminationManager
from trackmania_env.terminations.timeout_policy import TimeoutTerminationManager
from trackmania_env.terminations.noprogress_termination import NoProgressTerminationManager
from configs.config import TrainConfig

def get_termination_manager(cfg : TrainConfig) -> TerminationManager:
    """Returns instance of termination manager according to configuraion"""
    env_cfg = cfg.rl_env.env
    term_manager = cfg.rl_env.env.termination_manager
    if term_manager == "timeout_increase":
        return TimeoutTerminationManager(timeout = env_cfg.max_steps_until_reset, 
                                         ignore_stuck_for_n_steps_after_reset=env_cfg.ignore_stuck_for_n_steps_after_reset, 
                                         increase_timeout_intervals=env_cfg.increase_timeout_intervals, 
                                         new_timeouts=env_cfg.new_timeouts)
    elif term_manager == "no_progress":
        return NoProgressTerminationManager(timeout = env_cfg.max_steps_until_reset, 
                                         ignore_stuck_for_n_steps_after_reset=env_cfg.ignore_stuck_for_n_steps_after_reset, 
                                         terminate_after_steps_without_progress=env_cfg.terminate_after_steps_without_progress)
    
    else:
        raise ValueError(f"Termination-Manager '{term_manager}' not known.")