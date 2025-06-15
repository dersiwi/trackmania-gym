from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
from trackmania_env.utils.actionmap import ACTION_MAP


class TestEnvironment(TMNF_Single_Agent_Env):

    def __init__(self, command_queue, response_queue, obs_manager, position_buffer_size = 20, position_moved_threshold = 0.2, reset_mode = "respawn", reward_calculator = "basic", n_previous_actions = 10, ignore_stuck_for_n_steps_after_reset = 80, max_steps_before_reset = 10000, game_speed = 1):
        super().__init__(command_queue, response_queue, obs_manager, position_buffer_size, position_moved_threshold, reset_mode, reward_calculator, n_previous_actions, ignore_stuck_for_n_steps_after_reset, max_steps_before_reset, game_speed)

    def step(self, action):
        action = 0 #just go forward
        return super().step(action)
    
