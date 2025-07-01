from tminterface.structs import CheckpointData, SimStateData, CheckpointTime


class TerminationManager:

    def __init__(self, timeout : int, ignore_stuck_for_n_steps_after_reset : int):
        self.timeout : int = timeout

        self.ignore_stuck_for_n_steps_after_reset = ignore_stuck_for_n_steps_after_reset
        self.env = None

    def set_env(self, environment) -> None:
        """Sets environment for this Termination Manager"""
        from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
        self.env : TMNF_Single_Agent_Env = environment

    def calculate_terminations(self, observations : SimStateData) -> tuple[bool, bool, dict[str, bool]]:
        """Returns terminated, truncated for environment step."""
        raise NotImplementedError("Impement Own Class")

    def reset(self):
        pass


    # implemented termination terms, to use for subclasses 


    def calculate_timeout(self) -> bool:
        """Calculates if timeout has been reached"""
        return self.env.n_steps >= self.timeout
    

    def car_is_stuck(self) -> bool:
        stuck = False if self.env.n_steps < self.ignore_stuck_for_n_steps_after_reset else not self.env.position_buffer.moved_more_than_threshold(self.env.position_buffer_threshold)
        return stuck



