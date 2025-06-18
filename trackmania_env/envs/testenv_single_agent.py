from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
from trackmania_env.utils.actionmap import ACTION_MAP, get_reverse_action_map
import keyboard
from typing import Callable
import time
from matplotlib import pyplot as plt

class KEYS:
    UP = "nach-oben"
    DOWN = "nach-unten"
    LEFT = "nach-links"
    RIGHT = "nach-rechts"
    ESCAPE = "esc"

    @staticmethod
    def get_key_combo(left, right, accelerate, brake):
        combostring = ""
        if left:
            combostring += KEYS.LEFT + " : "
        if right:
                combostring += KEYS.RIGHT + " : "

        if accelerate:
                combostring += KEYS.UP + " : "

        if brake:
            combostring += KEYS.DOWN + " : "
        return combostring
    


class EnvironmentTestCallback():

    def __init__(self):
        pass

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        pass

    def _call_after_run(self):
        pass


class TestLinesightRewards(EnvironmentTestCallback):
    def __init__(self):
        super().__init__()
        self.v_x = []
        self.v_y = []
        self.v_z = []

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        velocity = info["velocity"]
        self.v_x.append(velocity[0])
        self.v_y.append(velocity[1])
        self.v_z.append(velocity[2])

    def _call_after_run(self):
        time = range(len(self.v_x))  

        plt.figure(figsize=(12, 6))

        plt.subplot(3, 1, 1)
        plt.plot(time, self.v_y, label='v_x')
        plt.ylabel('v_x')
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(time, self.v_y, label='v_y', color='orange')
        plt.ylabel('v_y')
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(time, self.v_z, label='v_z', color='green')
        plt.ylabel('v_z')
        plt.xlabel('Time Step')
        plt.grid(True)

        plt.tight_layout()
        plt.show()


class TestEnvironment(TMNF_Single_Agent_Env):

    def __init__(self, command_queue, response_queue, obs_manager, reward_calculator, env_cfg):
        super().__init__(command_queue, response_queue, obs_manager, reward_calculator, env_cfg=env_cfg)

        self.action_modifier : Callable = None
        self.step_while_doing_nothing = False
        """Variable for setp_with_manual_input. If not input was given, no (environment)-step is executed."""
        self.print_rewards_to_console = True
        """Variable for setp_with_manual_input. Prints each indiviaul reward term to the console after each environment step."""

        self.env_test_callback : EnvironmentTestCallback = None

    def _action_modifier_drive_forward(self, action : int) -> int:
        return 0
    
    def set_env_test_calback(self, env_test_callback : EnvironmentTestCallback):
        self.env_test_callback = env_test_callback
    
    def set_action_modifier(self, action_modifier : Callable) -> None:
        self.action_modifier = action_modifier

    def step(self, action):
        if not self.action_modifier == None:
            action = self.action_modifier(action)

        return super().step(action)
    
    def step_with_manual_input(self, time_between_actions : float = 0.012):
        REVERSE_ACTIONMAP = get_reverse_action_map()
        running = True
        no_actions_since_n_steps = 0 # indicates since how many steps no action was executed 
        while running:

            left, right, accelerate, brake = False, False, False, False

            if keyboard.is_pressed(KEYS.UP):
                accelerate = True

            if keyboard.is_pressed(KEYS.DOWN):
                brake = True

            if keyboard.is_pressed(KEYS.LEFT):
                left = True
            
            if keyboard.is_pressed(KEYS.RIGHT):
                right = True

            if keyboard.is_pressed(KEYS.ESCAPE):
                running = False

            
            reverse_action = (left, right, accelerate, brake)
            try:
                action_index = REVERSE_ACTIONMAP[reverse_action]
            except KeyError:
                print(f"Invalid action; key-combination : {KEYS.get_key_combo(*reverse_action)}")

            no_actions_since_n_steps = 0 if any(reverse_action) else no_actions_since_n_steps + 1

            if not self.step_while_doing_nothing and not any(reverse_action) and no_actions_since_n_steps >= 2:
                continue
            
            obs, reward, terminated, truncated, info = self.step(action_index)
            if terminated or truncated:
                super().reset()

            if not self.env_test_callback == None:
                self.env_test_callback._call_after_step(obs, reward, terminated, truncated, info)
            
            if self.print_rewards_to_console:
                print(info["rewards"])

            time.sleep(time_between_actions)

        if not self.env_test_callback == None:
            self.env_test_callback._call_after_run()
