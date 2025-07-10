"""
Custom Gymnasium Environment 
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
"""
from typing import Any,Dict,Tuple,Optional,List

import gymnasium as gym
import numpy as np
import logging
from collections import deque

from gymnasium import spaces
from queue import Queue
from tminterface.structs import CheckpointData, SimStateData, CheckpointTime

from game_interaction.process_wrapper import TMIProcessWrapper
from game_interaction.tminterface_commands import TMInterfaceCommands
from game_interaction.ipc_fields import IPCFields

from trackmania_env.utils.position_buffer import PositionBuffer
from trackmania_env.utils.actionmap import ACTION_MAP
from trackmania_env.observations.observation_manager import ObservationManager
from trackmania_env.rewards.reward_calculation import RewradCalculator
from trackmania_env.terminations.termination_manager import TerminationManager
from trackmania_env.utils.reference_line_manager import ReferenceLineManager
from trackmania_env.utils.random_respawn_manager import RandomRespawnManager

from configs.config import EnvConfig
from trackmania_env.utils.orientationless_random_respawn_manager import OrientationlessRespawnManager
import time


class TMNF_Single_Agent_Env(gym.Env):
    """The reinforcement learning environment for Trackmania Nations Forever"""

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}
    
    # init, step and reset have to be implemented for the class to be gym compatible
    def __init__(
            self,
            command_queue : Queue,
            response_queue : Queue,
            obs_manager : ObservationManager,
            reward_calculator : RewradCalculator,
            termination_manger : TerminationManager,
            reference_line: ReferenceLineManager,
            env_cfg : EnvConfig):
        
        """
        Initializes the custom Gymnasium environment.
        This constructor sets up the basic structure of the environment.
        As required by Gymnasium environments, it defines the action and observation spaces.

        Parameters  -- [WARNING] -- may be depricated; look at configuration in configs/rl_env/single_agent_env/env:
        ----------
        - coommand_queue            : Command-Queue used for sending commands to TMInterface process
        - response_queue            : Used for getting responsees from TMInterface process
        - obs_manager               : Processes raw-Observations aquired from TMInterface process, returns Observations given to Policy/FeatureExtractor
        - position_buffer_size      : Amount of positions that are tracked and from which the moved-distance is specified 
        - position_moved_threshold  : If position_buffer_size = n, it takes n steps in which the change in position has 
                to be less than position_moved_threshold for the environment to trigger a reset
        - reset_mode                : Specifies the mode how reset is execued. "respawn" uses game-respawn mechanic, "position" uses teleportation mode; ignores rotation
        - reward_calculator         : Instance of reward calcualtor used to calculate rewards in environment.
        - n_previous_actions        : tracks actions for this many steps. 
        - ignore_stuck_for_n_steps_after_reset : Ignores the position-buffer-reset-trigger for this many steps after reset. (Set to 1 if you dont want to use this)
        - game_speed                : sets speed of game, as defined in https://donadigo.com/tminterface/variables

        """
        self.n_prev_actions = env_cfg.n_previous_actions
        self.actions : deque = deque([(False,False,False,False)] * self.n_prev_actions, maxlen=self.n_prev_actions)
        """list of actions that may be stored later."""

        # variables for IPC communication
        self.command_queue = command_queue
        self.response_queue = response_queue
        self.__ipc_cmd_id = 0
        self.__ipc_timeout : int = 10

        self.logger = logging.getLogger(self.__class__.__name__)
        
        # reference line 
        self.reference_line = reference_line

        # variables used for resetting car(posiiton)
        self.start_position : list[float] = [0,0,0]
        self.__start_position_set : bool = False
        self.reset_mode = env_cfg.reset_mode
        
        self.position_buffer = PositionBuffer(env_cfg.position_buffer_size)
        self.position_buffer_threshold = env_cfg.position_moved_threshold

        self.rew_calculator = reward_calculator#get_reward_calculator(reward_calculator, self.position_buffer)
        self.rew_calculator.set_position_buffer(self.position_buffer)
        self.rew_calculator.set_reference_line(self.reference_line)
        
        self.obs_manager = obs_manager
        self.obs_manager.set_env(self)
        
        self.termination_manager = termination_manger
        self.termination_manager.set_env(self)

        self.random_respawn_manager = RandomRespawnManager(self.reference_line.reference_line)
        self.orientationless_respawn_manager : OrientationlessRespawnManager = None

        self.n_steps : int = 0
        self.total_steps : int = 0


        # define observation and action space for gym
        self.observation_space = obs_manager.get_observation_dict()
        self.action_space = gym.spaces.Discrete(len(ACTION_MAP))

        self.__send_command_to_process_wrapper(TMIProcessWrapper.IPCCommands.get_cmd_command(self.__ipc_cmd_id, 
                                                                                             TMInterfaceCommands.set_variable(TMInterfaceCommands.Variables.SPEED, 
                                                                                                                              value=env_cfg.game_speed)))
        self.__send_command_to_process_wrapper(TMIProcessWrapper.IPCCommands.get_cmd_command(self.__ipc_cmd_id, 
                                                                                                    TMInterfaceCommands.set_variable(TMInterfaceCommands.Variables.COUNTDOWN_SPEED, 
                                                                                                                                    value=env_cfg.countdown_speed)))
        # this is the defautl simstateData when the car first gets spawned. We will use this for the random reset
        self.default_ssD = None
        self.default_set = False
        self.first_reset = False
        
        self.waitforstep_timeout = env_cfg.waitforstep_timeout_in_s

    def set_respawn_manager(self, respawn_manager : OrientationlessRespawnManager):
        self.orientationless_respawn_manager = respawn_manager

    def _get_info(self,ssD) -> Dict[str,Any]:
        """Helper function for computing additional information (e.g. for debugging or logging)"""
        info = {}
        info["velocity"] = ssD.velocity
        info["position"] = ssD.position
        info["rotation_matrix"] = ssD.rotation_matrix
        info["dyna_rotation"] = ssD.dyna.current_state.rotation
        return info
    

    def __send_command_to_process_wrapper(self, command : dict[str, any], timeout = 10) -> dict[str, any]:
        """Sends a command to the process wrapper, asserts answer matches command and returns answer from process wrapper"""
        self.command_queue.put_nowait(command)
        response = self.response_queue.get(timeout=timeout)
        assert response[IPCFields.CMD_ID] == self.__ipc_cmd_id, f"Got unexepected command id from response. Expected {self.__ipc_cmd_id}, got : {response['cmd_id']}"
        self.__ipc_cmd_id += 1
        if not response[IPCFields.STATUS] == IPCFields.STATUS_OK:
            self.logger.error(f"Got error when executing command {command[IPCFields.CMD]}, with message : {response[IPCFields.ERROR]}")
        return response
    
    def __send_action(self, action : tuple[bool, bool, bool, bool]):
        self.__send_command_to_process_wrapper(TMIProcessWrapper.IPCCommands.get_act_command(self.__ipc_cmd_id, action))

    
    def __get_raw_obs(self) -> Dict:
        """Helper function to translate the the environment's state into an observation"""

        try:
            # request images, wait for response from process and check that cmd-id of response matches request-id.
            imgs_and_simstate = self.__send_command_to_process_wrapper(TMIProcessWrapper.IPCCommands.get_req_img_command(self.__ipc_cmd_id))
        except TimeoutError as t:
            self.logger.error(f"Timeout error while waiting for images: {t}")

        return imgs_and_simstate
    


    def __log_reset_reason(self, terminated_info : dict[str, bool]):
        for key in terminated_info:
            if terminated_info[key]:
                self.logger.info(f"Resetting environment because '{key}'")

    def determine_termination_trucation(self, idx, obs : SimStateData) -> tuple[bool, bool]:
        terminated = truncated = False

        return terminated, truncated

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
        raw_obs = self.__send_command_to_process_wrapper(TMIProcessWrapper.IPCCommands.step(self.__ipc_cmd_id, action))
        

        ssD : SimStateData = raw_obs[IPCFields.SIMSTATE]
        self.position_buffer.add(ssD.position)
        race_finished = ssD.player_info.race_finished

        # advance the reference line
        i, _, _ = self.reference_line.calculate_and_step_next_point(ssD.position)

        terminated, truncated, terminated_info = self.termination_manager.calculate_terminations(ssD)
        terminated_info["race_finished"] = race_finished
        terminated = terminated or race_finished
        
        if terminated or truncated:
            self.__log_reset_reason(terminated_info)

        if race_finished:
            self.__send_command_to_process_wrapper(TMIProcessWrapper.IPCCommands.prevent_simulation_finish(self.__ipc_cmd_id))

        info = self._get_info(ssD=ssD) 

        processed_obs, obs_info = self.obs_manager.get_observation(raw_obs)
        reward, reward_info = self.rew_calculator.calculate_reward(raw_obs, processed_obs, race_finished, terminated_info)

        info.update(obs_info)
        info["rewards"] = reward_info
        info["terminated"] = terminated
        info["truncated"] = truncated
        self.n_steps += 1
        self.total_steps += 1
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
        
        self.actions = deque([(False,False,False,False)] * self.n_prev_actions, maxlen=self.n_prev_actions)
        

        self.position_buffer.reset()
        self.rew_calculator.reset()
        self.reference_line.reset()
        self.termination_manager.reset()
        self.n_steps = 0

        # reset_car hsa to be done before! raw_obs is queried. 
        self.reset_car(None)    
        raw_obs = self.__get_raw_obs()

        self.reference_line.calculate_and_step_next_point(raw_obs[IPCFields.SIMSTATE].position)
        observation,obs_info = self.obs_manager.get_observation(raw_obs)
        info = self._get_info(ssD=raw_obs[IPCFields.SIMSTATE])
        info.update(obs_info)
        if not self.default_set:
            self.default_set = True
            self.default_ssD = raw_obs[IPCFields.SIMSTATE]

        if not self.first_reset:
            self.__send_command_to_process_wrapper(TMIProcessWrapper.IPCCommands.waitforstep(self.__ipc_cmd_id, self.waitforstep_timeout))
            self.first_reset = True

        return observation, info
    
    def reset_car(self, position : np.ndarray | list[float]):
        """Resets car depending on specified mode.
        Position : The position of the car in the current observation."""
        if self.reset_mode == "respawn"or self.reset_mode == "random_no_orientation":
            self.__respawn_car()
        elif self.reset_mode == "position" :
            if not self.__start_position_set:
                self.start_position = position
                self.__start_position_set = True
            self.__reset_car_position()
        else:
            raise ValueError(f"Mode '{self.reset_mode}' for car-resetting unknown.")
        
        if self.reset_mode == "random_no_orientation":
            # after car has been resettet to its starting position
            random_starting_pos, teleport = self.orientationless_respawn_manager.get_respawn_coordinates()
            if teleport:
                self.__send_command_to_process_wrapper(TMIProcessWrapper.IPCCommands.get_cmd_command(self.__ipc_cmd_id, 
                                                                                                    TMInterfaceCommands.teleport(random_starting_pos)))
                assert self.reference_line.next_point_idx == 0, f"Expected reference line to be resetted, but next_point_idx == {self.reference_line.next_point_idx}"
                nearest_idx = self.reference_line.locate_along_refline(random_starting_pos)
                self.reference_line.next_point_idx = max(nearest_idx - 3, 0) #put it a little behind, since calculation is performed again.
                self.logger.info(f"Set self.reference_line.next_point_idx = {self.reference_line.next_point_idx} after reset for position {random_starting_pos}.")
        
    def __respawn_car(self):
        """Respawns car by 'clicking' enter - uses internal game mechanic; also respawns in correct orientation"""
        self.__send_command_to_process_wrapper(TMIProcessWrapper.IPCCommands.get_cmd_command(self.__ipc_cmd_id, 
                                                                                    TMInterfaceCommands.key_action("press", "enter")))
    
    def __reset_car_position(self):
        """Executes teleportation command to stored car-position; ignores rotation and keeps current rotation of car."""
        self.__send_command_to_process_wrapper(TMIProcessWrapper.IPCCommands.get_cmd_command(self.__ipc_cmd_id, 
                                                                                    TMInterfaceCommands.teleport(self.start_position)))
    
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

    def random_reset(self,seed = None, options = None):
        super().reset(seed=seed, options=options) 
        if not self.default_set: self.reset()       
        raw_obs = self.__get_raw_obs()
        ssD = raw_obs[IPCFields.SIMSTATE] 
        reset_state = (self.random_respawn_manager.make_ssD_from_ref_point(ssD=self.default_ssD))
        self.__send_command_to_process_wrapper(TMIProcessWrapper.IPCCommands.rewind_state(self.__ipc_cmd_id, reset_state))