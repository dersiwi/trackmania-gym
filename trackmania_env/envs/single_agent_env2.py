"""
Custom Gymnasium Environment 
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
"""
from typing import Any,Dict,Tuple,Optional

import gymnasium as gym
import numpy as np
import time 
from queue import Queue
from tminterface.structs import CheckpointData, SimStateData, CheckpointTime


from game_interaction.tminterface2 import TMInterface, MessageType
from game_interaction.game_instance_manager2 import GameInstanceManager
from game_interaction.process_wrapper import TMIProcessWrapper

from trackmania_env.envs.actionmap import ACTION_MAP

import logging


class TMNF_Single_Agent_Env(gym.Env):
    """The reinforcement learning environment for Trackmania Nations Forever"""

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}
    
    # init, step and reset have to be implemented for the class to be gym compatible

    def __init__(
            self,
            img_width: int,
            img_height: int,
            observations_space: gym.spaces.Dict,
            gim: GameInstanceManager,
            command_queue : Queue,
            response_queue : Queue,
            color_channels : int = 3):
        """
        Initializes the custom Gymnasium environment.
        This constructor sets up the basic structure of the environment.
        As required by Gymnasium environments, it defines the action and observation spaces.
        We also define some other important varibales in order to communicate with TMInterface
        """
        self.img_width = img_width
        self.img_height = img_height
        self.color_channels= color_channels
        self.img_shape = (self.color_channels,img_height,img_width)
        
        self.gim = gim 
        self.observation_space = observations_space

        self.command_queue = command_queue
        self.response_queue = response_queue
        self.__ipc_cmd_id = 0
        self.__ipc_timeout : int = 10

        self.logger = logging.getLogger(self.__class__.__name__)
        
        """
        these are the inputs we will later give to the game engine ,
        copied from linesightrl /config_files/inputs_list.py
        """
        
        self.action_space = gym.spaces.Discrete(len(ACTION_MAP))
    
        
    def _get_info(self) -> Dict[str,Any]:
        """Helper function for computing additional information (e.g. for debugging or logging)"""
        info = {} 
        return info
    
    def _get_obs(self) -> Dict:
        """Helper function to translate the the environment's state into an observation"""

        try:
            # request images, wait for response from process and check that cmd-id of response matches request-id.
            self.command_queue.put(TMIProcessWrapper.IPCCommands.get_req_img_command(self.__ipc_cmd_id))
            imgs_and_simstate = self.response_queue.get(timeout=self.__ipc_timeout)
            assert imgs_and_simstate["cmd_id"] == self.__ipc_cmd_id
            self.__ipc_cmd_id += 1
        except TimeoutError as t:
            self.logger.error(f"Timeout error while waiting for images: {t}")


        image : np.ndarray = imgs_and_simstate["img"]
        game_states : SimStateData = imgs_and_simstate["sim_state"]
        sim_step : int = imgs_and_simstate["sim_step"]

        # TODO Issue #3 and #4

        gear = game_states.scene_mobil.engine.gear
        speed = game_states.scene_mobil.max_linear_speed
        burnout_state = game_states.scene_mobil.burnout_state  
        velocity = game_states.velocity
        observations = {
        "image": image,
        "gear": gear,
        "max_linear_speed": speed,
        "burnout_state": burnout_state,
        "velocity": velocity
        }
        return observations

    def step(self, action) -> Tuple[gym.spaces.Dict,float,bool,bool,Dict[str,Any]]:
        """
        The `step` function is a core component of the Gymnasium environment API and contains 
        the main game logic, including state transitions and reward calculations.
        This method applies the specified action to the environment, updates the 
        internal state accordingly, and returns the results of that action.

        Parameters:
            action (int): The action that should be performed

        Returns:
        observation (gym.spaces.Dict): The initial observation of the environment.
        reward (float): The scalar reward signal for the agent's action.
        terminated (bool): Whether the episode has ended because the task is successfully 
                           completed (e.g., agent reached the goal).
        truncated (bool): Whether the episode ended due to a time limit or other external cutoff.
        info (dict): A dictionary with additional information (e.g. for debugging or logging).
        """
        
        (left,right,accelerate,brake) = ACTION_MAP[action]

        # send the action via tminterface here
        self.command_queue.put(TMIProcessWrapper.IPCCommands.get_act_command(self.__ipc_cmd_id, (left,right,accelerate,brake)))
        res = self.response_queue.get(timeout=self.__ipc_timeout)
        assert res["cmd_id"] == self.__ipc_cmd_id
        self.__ipc_cmd_id += 1


        observations = self._get_obs()


        # TODO check if episode terminated and or tuncated, terminated if reached the goal, truncated means that a timelimit has been reached but MDP is not in a terminal state
        terminated = False
        truncated = False
        
        # TODO calculate reward
        reward = 0.
        
        # TODO also store some info for logging or debugging
        info = self._get_info() 
        
        return observations, reward, terminated, truncated, info
    
    def reset(self, seed = None, options = None)-> Tuple[gym.spaces.Dict,Dict[str,Any]]:
        """
        Resets the environment to start a new episode.

        Parameters:
            seed (int, optional): Seed for the random number generator to ensure 
                              deterministic behavior. Used by calling `super().reset(seed=seed)`.
            options (dict, optional): Additional options for resetting the environment. 
                                    These can be used to modify behavior at reset time.

        Returns:
            observation (gym.spaces.Dict): The initial observation of the environment.
            info (dict): Additional information that may be useful for debugging or analysis.
    
        This method is required by the Gymnasium API and is called at the beginning of each new episode.
        It ensures the environment starts in a consistent and valid state.
        """
        super().reset(seed=seed, options=options)
        """
        TODO here we can spawn the car (agent) on the same start positon or somewhere else 
        (one of the TM Youtube channels said that random spawning helps)
        """
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def render(self, mode = "human") -> Optional[np.array]:
        """
        This defines the render method. It supports:
            - mode="human": The environment is continuously rendered in the current 
            display or terminal, usually for human consumption.
            
            - mode="rgb_array": Return a single frame representing the current state
            of the environment. A frame is a np.ndarray with shape (x, y, 3) .
        """
        if mode == "human" :
            # TODO should we actually do here some render things ?
            return None
        if mode == "rgb_array":
            # TODO retun the actual frame
            return np.zeros(shape = self.img_shape)