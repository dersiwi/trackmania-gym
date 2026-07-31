

import keyboard
import time
from abc import abstractmethod
from typing import Callable

from trackmania_gym.game_interaction.ipc_command_sender import IPCommandSender

from trackmania_gym.trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env, ContinuousTMNF_Single_Agent_Env
from trackmania_gym.trackmania_env.utils.actionmap import REVERSE_ACTION_MAP

from trackmania_gym.trackmania_env.observations.observation_manager import ObservationManager 
from trackmania_gym.trackmania_env.rewards.reward_calculation import RewradCalculator
from trackmania_gym.trackmania_env.terminations.termination_manager import TerminationManager

from trackmania_gym.utils.keyboardwrapper import KEYS, KeyboardWrapper

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


    
class TestEnvironment():


    @staticmethod
    def get_testenv(env, platform, continuous_actions : bool):
        if continuous_actions:
            return ContinuousTestEnv(env, platform)
        else:
            return DiscreteTestEnv(env, platform)

    def __init__(self, environment : TMNF_Single_Agent_Env, platform = "windows"):

        self.platform = platform
        self.action_modifier : Callable = None
        self.step_while_doing_nothing = False
        """Variable for setp_with_manual_input. If not input was given, no (environment)-step is executed."""
        self.env_test_callback : list[TestEnvironmentCallback] = []

        self.keyboard = KeyboardWrapper.get_keyboardmodule(platform)
        self.env = environment

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

        return self.env.step(action)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    @abstractmethod
    def generate_action_from_keyboard(self) -> any:
        """Generates actions from keyboard inputs to give to the environment."""
        raise NotImplementedError("Implement in subclass-implementation")
    
    @abstractmethod
    def got_action_from_keyboard(self, generated_action) -> bool:
        """Returns true if the user put any action into the keyboard, false if not."""
        raise NotImplementedError("Implement in subclass-implementation")

    
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

            if self.keyboard.is_pressed(KEYS.RESET):
                for callback in self.env_test_callback:
                    callback.reset()
            
            action = self.generate_action_from_keyboard()
            got_action = self.got_action_from_keyboard(action)

            no_actions_since_n_steps = 0 if got_action else no_actions_since_n_steps + 1

            if not self.step_while_doing_nothing and not got_action and no_actions_since_n_steps >= 2:
                continue
            if self.keyboard.is_pressed(KEYS.ESCAPE):
                running = False            
            obs, reward, terminated, truncated, info = self.step(action)
            if terminated or truncated:
                self.reset()

            for cb in self.env_test_callback:
                cb._call_after_step(obs, reward, terminated, truncated, info)

            time.sleep(time_between_actions)

        for cb in self.env_test_callback:
            cb._call_after_run()


class DiscreteTestEnv(TestEnvironment):

    def __init__(self, environment, platform="windows"):
        super().__init__(environment, platform)

    def got_action_from_keyboard(self, generated_action):
        return not generated_action == REVERSE_ACTION_MAP[(False, False, False, False)]

    def generate_action_from_keyboard(self):
        left, right, accelerate, brake = False, False, False, False

        if self.keyboard.is_pressed(KEYS.UP):
            accelerate = True

        if self.keyboard.is_pressed(KEYS.DOWN):
            brake = True

        if self.keyboard.is_pressed(KEYS.LEFT):
            left = True
        
        if self.keyboard.is_pressed(KEYS.RIGHT):
            right = True

        

        if self.keyboard.is_pressed(KEYS.SHIFT):
            self.env.reset()

        reverse_action = (left, right, accelerate, brake)
        try:
            action_index = REVERSE_ACTION_MAP[reverse_action]
        except KeyError:
            print(f"Invalid action; key-combination : {KEYS.get_key_combo(*reverse_action)}")
            action_index = REVERSE_ACTION_MAP[(False, False, False, False)]

        return action_index
    

class ContinuousTestEnv(TestEnvironment):

    def __init__(self, environment, platform="windows"):
        super().__init__(environment, platform)

    def got_action_from_keyboard(self, generated_action):
        return not any(generated_action)

    def generate_action_from_keyboard(self):
        action = [0.0, 0.0, 0.0]
        if self.keyboard.is_pressed(KEYS.UP):
            action[1] = 1.0

        if self.keyboard.is_pressed(KEYS.DOWN):
            action[2] = 1.0

        if self.keyboard.is_pressed(KEYS.LEFT):
            action[0] = -1.0
        
        if self.keyboard.is_pressed(KEYS.RIGHT):
            action[0] = 1.0

        if self.keyboard.is_pressed(KEYS.SHIFT):
            self.env.reset()
        return action