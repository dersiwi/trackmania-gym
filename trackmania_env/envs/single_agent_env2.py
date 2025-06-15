"""
Custom Gymnasium Environment 
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
"""
from typing import Any,Dict,Tuple,Optional,List

import gymnasium as gym
import numpy as np
import logging
import torch

from gymnasium import spaces
from queue import Queue
from tminterface.structs import CheckpointData, SimStateData, CheckpointTime

from game_interaction.process_wrapper import TMIProcessWrapper
from game_interaction.tminterface_commands import TMInterfaceCommands
from game_interaction.ipc_fields import IPCFields

from trackmania_env.utils.position_buffer import PositionBuffer
from trackmania_env.utils.actionmap import ACTION_MAP
from trackmania_env.rewards.getrewards import get_reward_calculator
from trackmania_env.observations.observation_manager import ObservationManager
from collections import deque


from simstate_space_dict import simstate_space_dict



class TMNF_Single_Agent_Env(gym.Env):
    """The reinforcement learning environment for Trackmania Nations Forever"""

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}
    
    # init, step and reset have to be implemented for the class to be gym compatible
    def __init__(
            self,
            command_queue : Queue,
            response_queue : Queue,
            obs_manager : ObservationManager,
            position_buffer_size : int = 20,
            position_moved_threshold : float = 0.2,
            reset_mode : str = "respawn",
            reward_calculator : str = "basic",
            n_previous_actions : int = 10,
            max_steps_before_reset : int = 10000):
        """
        Initializes the custom Gymnasium environment.
        This constructor sets up the basic structure of the environment.
        As required by Gymnasium environments, it defines the action and observation spaces.
        We also define some other important varibales in order to communicate with TMInterface
        """
        self.n_prev_actions = n_previous_actions
        self.actions : deque = deque([(False,False,False,False)] * self.n_prev_actions, maxlen=self.n_prev_actions)
        """list of actions that may be stored later."""

        # variables for IPC communication
        self.command_queue = command_queue
        self.response_queue = response_queue
        self.__ipc_cmd_id = 0
        self.__ipc_timeout : int = 10

        self.logger = logging.getLogger(self.__class__.__name__)
        
        # variables used for resetting car(posiiton)
        self.start_position : list[float] = [0,0,0]
        self.__start_position_set : bool = False
        self.reset_mode = reset_mode
        
        self.position_buffer = PositionBuffer(position_buffer_size)
        self.position_buffer_threshold = position_moved_threshold

        self.rew_calculator = get_reward_calculator(reward_calculator, self.position_buffer)
        self.obs_manager = obs_manager
        self.obs_manager.set_env(self)

        self.max_steps_before_reset : int = max_steps_before_reset
        self.n_steps : int = 0

        # define observation and action space for gym
        self.observation_space = obs_manager.get_observation_dict()
        self.action_space = gym.spaces.Discrete(len(ACTION_MAP))
        
    def _get_info(self) -> Dict[str,Any]:
        """Helper function for computing additional information (e.g. for debugging or logging)"""
        info = {} 
        return info
    
    def _get_raw_obs(self) -> Dict:
        """Helper function to translate the the environment's state into an observation"""

        try:
            # request images, wait for response from process and check that cmd-id of response matches request-id.
            self.command_queue.put(TMIProcessWrapper.IPCCommands.get_req_img_command(self.__ipc_cmd_id))
            imgs_and_simstate = self.response_queue.get(timeout=self.__ipc_timeout)
            assert imgs_and_simstate[IPCFields.CMD_ID] == self.__ipc_cmd_id
            self.__ipc_cmd_id += 1
        except TimeoutError as t:
            self.logger.error(f"Timeout error while waiting for images: {t}")

        return imgs_and_simstate
    
    def _send_action(self, action : tuple[bool, bool, bool, bool]):
        self.command_queue.put(TMIProcessWrapper.IPCCommands.get_act_command(self.__ipc_cmd_id, action))
        res = self.response_queue.get(timeout=self.__ipc_timeout)
        assert res[IPCFields.CMD_ID] == self.__ipc_cmd_id
        self.__ipc_cmd_id += 1

    def __log_reset_reason(self, stuck, race_finished, timeout):
        if stuck:
            self.logger.info(f"Resetting environment because car is STUCk")
        elif race_finished:
            self.logger.info(f"Resttting environment because RACE FINISHED")
        elif timeout:
            self.logger.info(f"Resttting environment because TIMEOUT")  


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
        
        #store action internally and send via TMInterface
        action = ACTION_MAP[action]
        self.actions.append(action)
        self._send_action(action)


        raw_obs = self._get_raw_obs()
        ssD : SimStateData = raw_obs[IPCFields.SIMSTATE]
        self.position_buffer.add(ssD.position)
        race_finished = ssD.player_info.race_finished


        # TODO check if episode terminated and or tuncated, terminated if reached the goal, truncated means that a timelimit has been reached but MDP is not in a terminal state
        stuck = not self.position_buffer.moved_more_than_threshold(self.position_buffer_threshold)
        timeout = self.n_steps >= self.max_steps_before_reset
        terminated = stuck or race_finished or timeout
        self.__log_reset_reason(stuck, race_finished, timeout)
        

        truncated = False

        if race_finished:
            self.command_queue.put_nowait(TMIProcessWrapper.IPCCommands.prevent_simulation_finish())       

        
        
        # TODO also store some info for logging or debugging
        info = self._get_info() 

        processed_obs = self.obs_manager.get_observation(raw_obs)
        try:
            raw_obs["meters_advanced_along_centerline"] = self.obs_manager.distance_since_track_begin
            raw_obs["state_zone_center_coordinates_in_car_reference_system"] = self.obs_manager.state_zone_center_coordinates_in_car_reference_system
        except:
            # this only works if obs manager is lineisght 
            pass

        reward = self.rew_calculator.calculate_reward(raw_obs, race_finished, stuck)
        self.n_steps += 1
        return processed_obs, reward, terminated, truncated, info
    
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
        
        raw_obs = self._get_raw_obs()
        observation = self.obs_manager.get_observation(raw_obs)
        info = self._get_info()
        self.actions = deque([(False,False,False,False)] * self.n_prev_actions, maxlen=self.n_prev_actions)
        
        self.reset_car(raw_obs[IPCFields.SIMSTATE].position)
        self.position_buffer.reset()
        self.rew_calculator.reset()
        self.n_steps = 0

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
        assert response[IPCFields.CMD_ID] == self.__ipc_cmd_id, f"Got unexepected command id from response. Expected {self.__ipc_cmd_id}, got : {response['cmd_id']}"
        self.__ipc_cmd_id += 1

    
    def __reset_car_position(self):
        """Executes teleportation command to stored car-position; ignores rotation and keeps current rotation of car."""
        self.command_queue.put_nowait(TMIProcessWrapper.IPCCommands.get_cmd_command(self.__ipc_cmd_id, 
                                                                                    TMInterfaceCommands.teleport(self.start_position)))
        response = self.response_queue.get(10)
        assert response[IPCFields.CMD_ID] == self.__ipc_cmd_id, f"Got unexepected command id from response. Expected {self.__ipc_cmd_id}, got : {response['cmd_id']}"
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