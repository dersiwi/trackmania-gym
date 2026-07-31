from trackmania_gym.trackmania_env.terminations.termination_manager import TerminationManager
from trackmania_gym.trackmania_env.terminations.implemented_managers import NoProgressTerminationManager
from configs.config import TerminationManagerCfg

def get_termination_manager(termination_cfg : TerminationManagerCfg) -> TerminationManager:
    """Returns instance of termination manager according to configuraion"""
    name = termination_cfg.name
    if name == "timeout":
        return TerminationManager(timeout = termination_cfg.max_steps_until_reset, 
                                         ignore_stuck_for_n_steps_after_reset=termination_cfg.ignore_stuck_for_n_steps_after_reset)
    elif name == "no_progress":
        return NoProgressTerminationManager(timeout = termination_cfg.max_steps_until_reset, 
                                         ignore_stuck_for_n_steps_after_reset=termination_cfg.ignore_stuck_for_n_steps_after_reset, 
                                         terminate_after_steps_without_progress=termination_cfg.terminate_after_steps_without_progress)
    
    else:
        raise ValueError(f"Termination-Manager '{name}' not known.")