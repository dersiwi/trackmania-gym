from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
from trackmania_env.utils.actionmap import ACTION_MAP, get_reverse_action_map
import keyboard
from typing import Callable
import time
from matplotlib import pyplot as plt

class KEYS:
    """Enum for keys used in TestEnvironment."""
    UP = "nach-oben"
    DOWN = "nach-unten"
    LEFT = "nach-links"
    RIGHT = "nach-rechts"
    ESCAPE = "esc"

    @staticmethod
    def get_key_combo(left : bool, right : bool, accelerate : bool, brake : bool):
        """Translates the manual input of trackmania player into a string."""
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
    


class TestEnvironmentCallback():
    """TestEnviornmentCallbacks are used to track, log, do whatever with data obtained by an environment per setp."""

    def __init__(self):
        pass

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        """This method is called by TestEnvironment.step_with_manual_input(), after everytime this method executes
        an environment step of the underlying environment."""
        pass

    def _call_after_run(self):
        """This method is called by TestEnvironment.step_with_manual_input(), after the main-loop has been executed via `esc`."""
        pass

class PrintRewardsToConsole(TestEnvironmentCallback):

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        print(info["rewards"])



class TestLinesightRewards(TestEnvironmentCallback):
    """Tacks vx, vy, vz and plots them after run."""
    
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
        self.env_test_callback : list[TestEnvironmentCallback] = []

    def _action_modifier_drive_forward(self, action : int) -> int:
        """Action-modifier which is valid option to set for self.action_modifier."""
        return 0
    
    def add_env_test_calback(self, env_test_callback : TestEnvironmentCallback):
        """Adds testenvironment-callback used in self.step_with_manual_input()"""
        self.env_test_callback.append(env_test_callback)
    
    def set_action_modifier(self, action_modifier : Callable) -> None:
        self.action_modifier = action_modifier

    def step(self, action):
        """Calls super().step(action). If no self.action_modifier (e.g. self._action_moifier_drive_forawrd) is defined."""
        if not self.action_modifier == None:
            action = self.action_modifier(action)

        return super().step(action)
    
    def step_with_manual_input(self, time_between_actions : float = 0.012):
        """This method enables to send manual inputs to the TMNF_Single_Agent_Env; basically simulating you playing the game with extra steps.
        In order to do something with date coming from the environment after each-step, you can define your own TestEnvironmentCallback-class.
        Implement the methods _call_after_step(obs, rew, terminated, trucated, infos) and _call_after_run(). 
        Then, before callinth this method add them to this class via this.add_env_test_callback(). 

        After you start this method the main loop starts:
        ``` 
        while running:
            1. press key 
            2. complete environment step
            3. call _call_after_step for each added callback
        4. call _call_after_run
        ```
        The run ends, if you press `esc`.

        """
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

            for cb in self.env_test_callback:
                cb._call_after_step(obs, reward, terminated, truncated, info)

            time.sleep(time_between_actions)

        for cb in self.env_test_callback:
            cb._call_after_run()
