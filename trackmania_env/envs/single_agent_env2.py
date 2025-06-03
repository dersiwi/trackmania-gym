"""
Custom Gymnasium Environment 
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
"""
from typing import Any,Dict,Tuple,Optional,List

import gymnasium as gym
from gymnasium.spaces import Box,Discrete 
from numpy import uint8,int32,float32,inf
import numpy as np
import time 
from queue import Queue
from tminterface.structs import CheckpointData, SimStateData, CheckpointTime

from game_interaction.process_wrapper import TMIProcessWrapper
from game_interaction.tminterface_commands import TMInterfaceCommands

from trackmania_env.envs.position_buffer import PositionBuffer
from trackmania_env.envs.actionmap import ACTION_MAP
from trackmania_env.envs.reward_calculation import RewradCalculator

import logging

class TMNF_Single_Agent_Env(gym.Env):
    """The reinforcement learning environment for Trackmania Nations Forever"""

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}
    
    # init, step and reset have to be implemented for the class to be gym compatible

    def __init__(
            self,
            command_queue : Queue,
            response_queue : Queue,
            position_buffer_size : int = 20,
            position_moved_threshold : float = 0.2,
            reset_mode : str = "respawn",
            observation_space = gym.spaces.Dict({
                "image" : gym.spaces.Box(low=0, high=255, shape=(3,100,100), dtype=np.uint8),
                "velocity": Box(-inf, inf, (3,), float32),
                "yaw_pitch_roll": Box(-inf, inf, (3,), float32) ,
                "position": Box(-inf, inf, (3,), float32),
                "scene_mobil_field.engine_field.gear": Box(-inf, inf, (), float32),
            }),):
        """
        Initializes the custom Gymnasium environment.
        This constructor sets up the basic structure of the environment.
        As required by Gymnasium environments, it defines the action and observation spaces.
        We also define some other important varibales in order to communicate with TMInterface
        """  
        self.observation_space = observation_space

        self.command_queue = command_queue
        self.response_queue = response_queue
        self.__ipc_cmd_id = 0
        self.__ipc_timeout : int = 10



        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.action_space = gym.spaces.Discrete(len(ACTION_MAP))
        self.SimStateData = None
        
        self.actions = []
        """list of actions that may be stored later."""


        # variables used for resetting car(posiiton)
        self.start_position : list[float] = [0,0,0]
        self.__start_position_set : bool = False
        self.reset_mode = reset_mode
        
        self.position_buffer = PositionBuffer(position_buffer_size)
        self.position_buffer_threshold = position_moved_threshold

        self.rew_calculator = RewradCalculator(self.position_buffer)
        
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
        self.SimStateData = game_states
        # TODO Issue #3
        # the wrapper classes will do the filtering

        observations = {
        "image": image,
        "SimStateData": game_states,
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

        self.actions.append((left,right,accelerate,brake))

        # send the action via tminterface here
        self.command_queue.put(TMIProcessWrapper.IPCCommands.get_act_command(self.__ipc_cmd_id, (left,right,accelerate,brake)))
        res = self.response_queue.get(timeout=self.__ipc_timeout)
        assert res["cmd_id"] == self.__ipc_cmd_id
        self.__ipc_cmd_id += 1


        observations = self._get_obs()
        ssD : SimStateData = observations["SimStateData"]

        self.position_buffer.add(ssD.position)
        race_finished = ssD.player_info.race_finished


        # TODO check if episode terminated and or tuncated, terminated if reached the goal, truncated means that a timelimit has been reached but MDP is not in a terminal state
        terminated = not self.position_buffer.moved_more_than_threshold(self.position_buffer_threshold)
        truncated = False
        

        reward = self.rew_calculator.calculate_reward(observations)
        
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
        self.actions = []
        
        self.reset_car(observation["SimStateData"].position)
        self.position_buffer.reset()
        self.rew_calculator.reset()

        return observation, info
    
    def reset_car(self, position : np.ndarray | list[float]):
        """Resets car depending on specified mode.
        Position : The position of the car in the current observation."""
        if self.reset_mode == "respawn":
            self.__respawn_car()
        elif self.reset_mode == "position":
            if not self.__start_position_set:
                self.start_position = position
                self.__start_position_set = True
            self.__reset_car_position()
        else:
            raise ValueError(f"Mode '{self.reset_mode}' for car-resetting unknown.")
    
    def __respawn_car(self):
        """Respawns car by 'clicking' enter - uses internal game mechanic; also respawns in correct orientation"""
        self.command_queue.put_nowait(TMIProcessWrapper.IPCCommands.get_cmd_command(self.__ipc_cmd_id, 
                                                                                    TMInterfaceCommands.key_action("press", "enter")))
        response = self.response_queue.get(10)
        assert response["cmd_id"] == self.__ipc_cmd_id, f"Got unexepected command id from response. Expected {self.__ipc_cmd_id}, got : {response['cmd_id']}"
        self.__ipc_cmd_id += 1

    
    def __reset_car_position(self):
        """Executes teleportation command to stored car-position; ignores rotation and keeps current rotation of car."""
        self.command_queue.put_nowait(TMIProcessWrapper.IPCCommands.get_cmd_command(self.__ipc_cmd_id, 
                                                                                    TMInterfaceCommands.teleport(self.start_position)))
        response = self.response_queue.get(10)
        assert response["cmd_id"] == self.__ipc_cmd_id, f"Got unexepected command id from response. Expected {self.__ipc_cmd_id}, got : {response['cmd_id']}"
        self.__ipc_cmd_id += 1
    
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
        

    def store_actions(self, filename : str):
        with open(filename, "w") as file:
            file.write(f"{self.actions}\n")