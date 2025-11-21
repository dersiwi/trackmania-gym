from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
from trackmania_env.utils.actionmap import REVERSE_ACTION_MAP

import keyboard
from typing import Callable
import time
from pynput.keyboard import Key, Listener,KeyCode

from queue import Queue

from trackmania_env.observations.observation_manager import ObservationManager 
from trackmania_env.rewards.reward_calculation import RewradCalculator
from trackmania_env.terminations.termination_manager import TerminationManager
from trackmania_env.utils.reference_line_manager import ReferenceLineManager

class TestEnvironmentCallback():
    """TestEnviornmentCallbacks are used to track, log, do whatever with data obtained by an environment per setp."""

    def __init__(self):
        self.n_step = 0
        """Counts environment-steps aka. how often _call_after_step was called."""

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        """This method is called by TestEnvironment.step_with_manual_input(), after everytime this method executes
        an environment step of the underlying environment."""
        pass

    def _call_after_run(self):
        """This method is called by TestEnvironment.step_with_manual_input(), after the main-loop has been executed via `esc`."""
        pass

    def reset(self):
        """Resets the callback, if the user presses 'r'"""
        pass

class KEYS:
    """Enum for keys used in TestEnvironment."""
    UP = "nach-oben"
    DOWN = "nach-unten"
    LEFT = "nach-links"
    RIGHT = "nach-rechts"
    ESCAPE = "esc"
    SHIFT = "shift"
    RESET = "k"

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
    
class TestEnvironment(TMNF_Single_Agent_Env):

    def __init__(self, 
                 command_queue: Queue, 
                 response_queue: Queue, 
                 obs_manager: ObservationManager, 
                 reward_calculator: RewradCalculator, 
                 termination_manger: TerminationManager, 
                 reference_line: ReferenceLineManager, 
                 reset_mode: str, 
                 n_previous_actions: int, 
                 position_buffer_size: int, 
                 position_moved_threshold: float, 
                 ignore_stuck_for_n_steps_after_reset: int, 
                 game_speed: int, 
                 countdown_speed: int, 
                 waitforstep_timeout_in_s: float, 
                 startposition_accuracy_threshold: float, 
                 gamma: float, 
                 platform = "windows",
                 **kwargs):

        super().__init__(command_queue, response_queue, obs_manager, reward_calculator, termination_manger, reference_line, reset_mode, n_previous_actions, position_buffer_size, position_moved_threshold, ignore_stuck_for_n_steps_after_reset, game_speed, countdown_speed, waitforstep_timeout_in_s, startposition_accuracy_threshold, gamma, **kwargs)

        self.platform = platform
        self.action_modifier : Callable = None
        self.step_while_doing_nothing = False
        """Variable for setp_with_manual_input. If not input was given, no (environment)-step is executed."""
        self.env_test_callback : list[TestEnvironmentCallback] = []

        self.keyboard = keyboard if platform == "windows" else LinuxKeyboardWrapper()

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
        running = True
        no_actions_since_n_steps = 0 # indicates since how many steps no action was executed 
        while running:

            left, right, accelerate, brake = False, False, False, False

            if self.keyboard.is_pressed(KEYS.UP):
                accelerate = True

            if self.keyboard.is_pressed(KEYS.DOWN):
                brake = True

            if self.keyboard.is_pressed(KEYS.LEFT):
                left = True
            
            if self.keyboard.is_pressed(KEYS.RIGHT):
                right = True

            if self.keyboard.is_pressed(KEYS.ESCAPE):
                running = False

            if self.keyboard.is_pressed(KEYS.SHIFT):
                #super().random_reset()
                pass

            if self.keyboard.is_pressed(KEYS.RESET):
                for callback in self.env_test_callback:
                    callback.reset()
            
            reverse_action = (left, right, accelerate, brake)
            try:
                action_index = REVERSE_ACTION_MAP[reverse_action]
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

class LinuxKeyboardWrapper:
    def __init__(self):
        self.key_map = {
            "nach-oben": Key.up,
            "nach-unten": Key.down,
            "nach-links": Key.left,
            "nach-rechts": Key.right,
            "esc": Key.esc,
            "shift": Key.shift,
            "k" : KeyCode.from_char(KEYS.RESET)
        }

        # Keep track of currently pressed keys
        self.pressed_keys = set()

        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        self.pressed_keys.add(key)

    def on_release(self, key):
        self.pressed_keys.discard(key)

    def is_pressed(self, key_str):
        pynput_key = self.key_map.get(key_str)
        if pynput_key is None:
            raise ValueError(f"Key '{key_str}' is not mapped in LinuxKeyboardWrapper.")
        return pynput_key in self.pressed_keys
