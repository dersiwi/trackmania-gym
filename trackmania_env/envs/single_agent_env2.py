"""
Custom Gymnasium Environment for TMNF using InterprocessCommunication to talk to TMIProcessWrapper.
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
"""
from __future__ import annotations
from typing import Any,Dict,Tuple,Optional,List

import gymnasium as gym
import numpy as np
import logging
import time

from collections import deque

from queue import Queue, Empty

from tminterface.structs import SimStateData

from game_interaction.ipc_fields import IPCCommands
from game_interaction.tminterface_commands import TMInterfaceCommands
from game_interaction.ipc_fields import IPCFields

from trackmania_env.observations.observation_manager import ObservationManager 
from trackmania_env.rewards.reward_calculation import RewradCalculator
from trackmania_env.terminations.termination_manager import TerminationManager
from trackmania_env.utils.reference_line_manager import ReferenceLineManager
from trackmania_env.utils.random_respawn_manager import RandomRespawnManager
from trackmania_env.utils.orientationless_random_respawn_manager import OrientationlessRespawnManager
from trackmania_env.utils.return_tracker import ReturnTracker
from trackmania_env.utils.position_buffer import PositionBuffer
from trackmania_env.utils.actionmap import ACTION_MAP, ActionMode
from trackmania_env.envs.info import EnvironmentInfo


class TMICommunicationFaildException(Exception):
    """Custom exception for the error repeatedely encountered during training."""
    def __init__(self, n_tries: str, message: str = None):
        if message is None:
            message = f"Process Wrapper did not handle command correctly. Tried {n_tries} times."
        super().__init__(message)


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
            track : str,
            reset_mode:str,
            n_previous_actions:int,
            position_buffer_size:int,
            position_moved_threshold:float,
            ignore_stuck_for_n_steps_after_reset:int,
            game_speed:int,
            countdown_speed:int,
            waitforstep_timeout_in_s:float,
            startposition_accuracy_threshold:float,
            gamma:float,
            is_discrete:bool = True,
            **kwargs):
        
        """
        Initializes the custom Gymnasium environment.
        This constructor sets up the basic structure of the environment.
        As required by Gymnasium environments, it defines the action and observation spaces.
        -- [WARNING] -- may be depricated; look at configuration in configs/rl_env/single_agent_env/env

        Args:
            coommand_queue (Queue)  : Command-Queue used for sending commands to TMInterface process
            response_queue (Queue)  : Used for getting responsees from TMInterface process
            obs_manager (ObservationManager)    : Processes raw-Observations aquired from TMInterface process, returns Observations given to Policy/FeatureExtractor
            reward_calculator (RewradCalculator): Instance of reward calcualtor used to calculate rewards in environment.

            position_buffer_size (int)          : Amount of positions that are tracked and from which the moved-distance is specified 
            position_moved_threshold (float)    : If position_buffer_size = n, it takes n steps in which the change in position has 
                to be less than position_moved_threshold for the environment to trigger a reset
            reset_mode  (str)       : Specifies the mode how reset is execued. "respawn" uses game-respawn mechanic, "position" uses teleportation mode; ignores rotation
            n_previous_actions (int): tracks actions for this many steps. 
            ignore_stuck_for_n_steps_after_reset (int): Ignores the position-buffer-reset-trigger for this many steps after reset. (Set to 1 if you dont want to use this)
            game_speed (int)        : sets speed of game, as defined in https://donadigo.com/tminterface/variables

        """
        self.n_prev_actions = n_previous_actions 
        self.actions : deque = deque([(False,False,False,False)] * self.n_prev_actions, maxlen=self.n_prev_actions)
        """list of actions that may be stored later."""

        # variables for IPC communication
        self.command_queue = command_queue
        self.response_queue = response_queue
        self.ipc_cmd_id = 0
        """Command id for interprocess communication"""
        self.__ipc_timeout : int = 10

        self.logger = logging.getLogger(self.__class__.__name__)

        self.track = track
        self.reference_line : ReferenceLineManager = ReferenceLineManager.load_dynamically(self.track)

        # variables used for resetting car(posiiton)
        self.start_position : np.ndarray = None
        self.reset_mode = reset_mode
        self.start_position_set : bool = False
        
        self.position_buffer = PositionBuffer(position_buffer_size)
        self.position_buffer_threshold = position_moved_threshold

        self.ignore_stuck_for_n_steps_after_reset = ignore_stuck_for_n_steps_after_reset

        self.rew_calculator = reward_calculator
        self.obs_manager = obs_manager
        self.termination_manager = termination_manger

        self.random_respawn_manager = RandomRespawnManager(self.reference_line.reference_line)
        self.orientationless_respawn_manager : OrientationlessRespawnManager = None

        self.n_steps : int = 0
        self.total_steps : int = 0

        self.__wrong_starting_position = 0
        """Counts how often the starting position was missed. Then teleports it slightly infornt of the currents tarting position after too many misses."""

        self.rt = ReturnTracker(length = self.termination_manager.timeout, gamma = gamma)


        # define observation and action space for gym
        self.observation_space = obs_manager.get_observation_space()
        self.action_space = gym.spaces.Discrete(len(ACTION_MAP))
        self.is_discrete = is_discrete

        self._send_command_to_process_wrapper(IPCCommands.get_cmd_command(self.ipc_cmd_id, 
                                                                                             TMInterfaceCommands.set_variable(TMInterfaceCommands.Variables.SPEED, 
                                                                                                                              value=game_speed)))
        self._send_command_to_process_wrapper(IPCCommands.get_cmd_command(self.ipc_cmd_id, 
                                                                                                TMInterfaceCommands.set_variable(TMInterfaceCommands.Variables.COUNTDOWN_SPEED, 
                                                                                                                                value=countdown_speed)))
        # this is the defautl simstateData when the car first gets spawned. We will use this for the random reset
        self.default_ssD = None
        self.default_set = False
        self.first_reset = False
        
        self.waitforstep_timeout = waitforstep_timeout_in_s

        self.startposition_accuracy_threshold = startposition_accuracy_threshold

        self.info = {}

        # NOTE: this always has to come last. 
        # setting the environment for all of the managers who need it
        self.obs_manager.set_env(self)
        self.rew_calculator.set_env(self)
        self.termination_manager.set_env(self)

    def set_respawn_manager(self, respawn_manager : OrientationlessRespawnManager):
        self.orientationless_respawn_manager = respawn_manager

    def _get_info(self,ssD:SimStateData) -> Dict[str,Any]:
        """Helper function for computing additional information (e.g. for debugging or logging)"""
        info = {}
        info[EnvironmentInfo.DISPLAY_SPEED] = ssD.display_speed
        info[EnvironmentInfo.GAS] = ssD.scene_mobil.input_gas
        info[EnvironmentInfo.LAST_HAS_ANY_LATERAL_CONTACT_TIME] = ssD.scene_mobil.last_has_any_lateral_contact_time
        info[EnvironmentInfo.VELOCITY] = ssD.velocity
        info[EnvironmentInfo.POSITION] = ssD.position
        info[EnvironmentInfo.ROTATION_MATRIX] = ssD.rotation_matrix
        info[EnvironmentInfo.DYNA_ROTATION] = ssD.dyna.current_state.rotation
        info[EnvironmentInfo.NEXT_REFLINE_IDX] = self.reference_line.next_point_idx
        return info
    

    def _send_command_to_process_wrapper(self, command : dict[str, any], timeout = 10) -> dict[str, any]:
        """Sends a command to the process wrapper, asserts answer matches command and returns answer from process wrapper"""
        max_commnd_sending_attempts = 1
        attempt = 0
        while attempt <= max_commnd_sending_attempts:
            attempt += 1
            try:
                self.command_queue.put_nowait(command)
                response = self.response_queue.get(timeout=timeout)
                assert response[IPCFields.CMD_ID] == self.ipc_cmd_id, f"Got unexepected command id from response. Expected {self.ipc_cmd_id}, got : {response['cmd_id']}"
                self.ipc_cmd_id += 1
                if not response[IPCFields.STATUS] == IPCFields.STATUS_OK:
                    self.logger.error(f"Got error when executing command {command[IPCFields.CMD]}, with message : {response[IPCFields.ERROR]}")
                    raise AttributeError()
                return response
            except Empty as e:
                self.logger.error(f"Tried sending commmand '{command[IPCFields.CMD_ID]}' {attempt} times and got no response; queue is empty.")
            except TimeoutError as tr:
                self.logger.error(f"Tried sending commmand '{command[IPCFields.CMD_ID]}' {attempt} times and got timeout error.")
            except AttributeError as at:
                self.logger.error(f"Tried sending commmand '{command[IPCFields.CMD_ID]}' {attempt} times and Attribute Error.")

            time.sleep(0.5) #<- waiting period between command-sends.

        raise TMICommunicationFaildException(n_tries=attempt, message="Were not able to send command even after multiple tries.") # <- If this happens; the responding end most likely crashed.


    def __get_raw_obs(self) -> Dict:
        """Helper function to translate the the environment's state into an observation"""

        try:
            # request images, wait for response from process and check that cmd-id of response matches request-id.
            imgs_and_simstate = self._send_command_to_process_wrapper(IPCCommands.get_req_img_command(self.ipc_cmd_id))
        except TimeoutError as t:
            self.logger.error(f"Timeout error while waiting for images: {t}")

        imgs_and_simstate[IPCFields.ACTION] = self.actions[-1] # TODO this is pretty ugly but current the easiest and best option to inlcude the action 
        return imgs_and_simstate
    

    def __log_reset_reason(self, terminated_info : dict[str, bool]):
        for key in terminated_info:
            if terminated_info[key]:
                self.logger.info(f"Resetting environment because '{key}'")

    def determine_termination_trucation(self, idx, obs : SimStateData) -> tuple[bool, bool]:
        terminated = truncated = False
        return terminated, truncated
    
    def request_map(self, trackname : str) -> None:
        """Sends a map-request command to the porcess-wrapper. Furthermore reloads referenceline and random-respawn manager. Also 
        sets mechanism to find new start-position."""
        self._send_command_to_process_wrapper(IPCCommands.get_cmd_command(self.ipc_cmd_id, TMInterfaceCommands.map(trackname)))
        self.track = trackname
        self.reference_line : ReferenceLineManager = ReferenceLineManager.load_dynamically(trackname)
        self.random_respawn_manager = RandomRespawnManager(self.reference_line.reference_line)
        self.start_position = None
        self.start_position_set = False

    def _get_action(self, action) -> tuple[bool]:
        """This method takes the action as given by the learner (i.e. the agent) and translates it into something
        the ProcessWrapper can work with. In the discrete case this is a tuple of booleans:
            (left, right, accelerate, brake)"""
        return ACTION_MAP[action]

    def step(self, action) -> Tuple[gym.spaces.Dict,float,bool,bool,Dict[str,Any]]:
        """
        The `step` function is a core component of the Gymnasium environment API and contains 
        the main game logic, including state transitions and reward calculations.
        This method applies the specified action to the environment, updates the 
        internal state accordingly, and returns the results of that action.

        Args:
            action (int): The action that should be performed

        Returns:
            Tuple (gym.spaces.Dict,float,bool,bool,Dict[str,Any]) : As per gymnasium-interface specified.
                - observation (gym.spaces.Dict): The initial observation of the environment
                - reward (float)      : The scalar reward signal for the agent's action
                - terminated (bool)   : Whether the episode has ended because the task is successfully completed (e.g., agent reached the goal)
                - truncated (bool)    : Whether the episode ended due to a time limit or other external cutoff
                - info (dict)         : A dictionary with additional information (e.g. for debugging or logging).
        """
        
        #store action internally and send via TMInterface
        action = self._get_action(action)
        self.actions.append(action)
        raw_obs = self._send_command_to_process_wrapper(IPCCommands.step(self.ipc_cmd_id, action))
        raw_obs[IPCFields.ACTION] = action 

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

        info = self._get_info(ssD=ssD) 

        processed_obs, obs_info = self.obs_manager.get_observation(raw_obs)
        reward, reward_info = self.rew_calculator.calculate_reward(raw_obs, processed_obs, race_finished, terminated_info)

        self.rt.add_reward(reward)
        

        info.update(obs_info)
        info["rewards"] = reward_info
        info["terminated"] = terminated
        info["truncated"] = truncated
        info["action"] = action
        if terminated or truncated:
            info[ReturnTracker.LOG_NAME] = self.rt.get_return()

        self.n_steps += 1
        self.total_steps += 1
        
        # # Ensures reward is a scalar to prevent NumPy DeprecationWarnings
        reward = reward.item()

        return processed_obs, reward, terminated, truncated, info
    
    def reset(self, seed = None, options = None)-> Tuple[gym.spaces.Dict,Dict[str,Any]]:
        """
        Resets the environment to start a new episode. This method is required by the Gymnasium API and is called at the beginning of each new episode.
        It ensures the environment starts in a consistent and valid state.

        Args:
            seed (int, optional): Seed for the random number generator to ensure 
                              deterministic behavior. Used by calling `super().reset(seed=seed)`.
            options (dict, optional): Additional options for resetting the environment. 
                                    These can be used to modify behavior at reset time.

        Returns:
            Tuple (gym.spaces.Dict,Dict[str,Any]) : As per gymnasium-interface specified.
            
                - observation (gym.spaces.Dict): The initial observation of the environment.
                - info (dict): Additional information that may be useful for debugging or analysis.
        """
        super().reset(seed=seed, options=options)
        
        self.actions = deque([(False,False,False,False)] * self.n_prev_actions, maxlen=self.n_prev_actions)
        
        self.obs_manager.reset()
        self.position_buffer.reset()
        self.rew_calculator.reset()
        self.reference_line.reset()
        self.termination_manager.reset()
        self.rt.reset()
        self.n_steps = 0

        # reset_car hsa to be done before! raw_obs is queried. 
        self.reset_car(None)
        raw_obs = self.__get_raw_obs()

        if self.start_position_set:
            tries = 0
            """Sometimes the raw obs queried after reset are not actually the ones after reset - therefore this primitive check is implemeneted which checks that the first starting position;
            which is recorded before any step was made somewhat matches the position after a reset; if so, collects more obs."""
            while np.linalg.norm(raw_obs[IPCFields.SIMSTATE].position - self.start_position) > self.startposition_accuracy_threshold and tries < 10:
                tries += 1
                self.logger.warning(f"Start position did not match obs posiiton. Waiting until resetted. Tries : {tries}")
                time.sleep(0.005)
                raw_obs = self.__get_raw_obs()
                self.__wrong_starting_position += 1
        
        if self.__wrong_starting_position >= 20:
            self._teleport_car_slightly()


        self.reference_line.calculate_and_step_next_point(raw_obs[IPCFields.SIMSTATE].position)
        observation,obs_info = self.obs_manager.get_observation(raw_obs)
        info = self._get_info(ssD=raw_obs[IPCFields.SIMSTATE])
        info.update(obs_info)
        if not self.default_set:
            self.default_set = True
            self.default_ssD = raw_obs[IPCFields.SIMSTATE]

        if not self.first_reset:
            self._send_command_to_process_wrapper(IPCCommands.waitforstep(self.ipc_cmd_id, self.waitforstep_timeout))
            self.first_reset = True
        if not self.start_position_set:
            self.start_position = np.array(raw_obs[IPCFields.SIMSTATE].position)
            self.start_position_set = True
            

        return observation, info
    
    def _teleport_car_slightly(self):
        """The idea is that by slightly teleporting the car, the game can reset itself. But thats just a hunch."""
        telepos = np.array([0.1, 0.0, 0.0])
        if self.start_position is not None:
            telepos += self.start_position
        self.logger.info(f"Sending command to teleport car to : {telepos}")
        self._send_command_to_process_wrapper(IPCCommands.get_cmd_command(self.ipc_cmd_id, TMInterfaceCommands.teleport(telepos )))
        self.__wrong_starting_position = 0
    
    def reset_car(self, position : np.ndarray | list[float]):
        """Resets car depending on specified mode.
        Position : The position of the car in the current observation."""

        if self.reset_mode == "respawn"or self.reset_mode == "random_no_orientation":
            self.__respawn_car()
        elif self.reset_mode == "position" :
            self.__reset_car_position()
        else:
            raise ValueError(f"Mode '{self.reset_mode}' for car-resetting unknown.")
        
        if self.reset_mode == "random_no_orientation":
            # after car has been resettet to its starting position
            random_starting_pos, teleport = self.orientationless_respawn_manager.get_respawn_coordinates()
            if teleport:
                self._send_command_to_process_wrapper(IPCCommands.get_cmd_command(self.ipc_cmd_id, 
                                                                                                    TMInterfaceCommands.teleport(random_starting_pos)))
                assert self.reference_line.next_point_idx == 0, f"Expected reference line to be resetted, but next_point_idx == {self.reference_line.next_point_idx}"
                nearest_idx = self.reference_line.locate_along_refline(random_starting_pos)
                self.reference_line.next_point_idx = max(nearest_idx - 3, 0) #put it a little behind, since calculation is performed again.
                self.logger.info(f"Set self.reference_line.next_point_idx = {self.reference_line.next_point_idx} after reset for position {random_starting_pos}.")
        
    def __respawn_car(self):
        """Respawns car by 'clicking' enter - uses internal game mechanic; also respawns in correct orientation"""
        self._send_command_to_process_wrapper(IPCCommands.get_cmd_command(self.ipc_cmd_id, 
                                                                                    TMInterfaceCommands.key_action("press", "enter")))
    
    def __reset_car_position(self):
        """Executes teleportation command to stored car-position; ignores rotation and keeps current rotation of car."""
        self._send_command_to_process_wrapper(IPCCommands.get_cmd_command(self.ipc_cmd_id, 
                                                                                    TMInterfaceCommands.teleport(self.start_position)))
    
    def render(self, mode = "rgb_array") -> Optional[np.array]:
        """
        This defines the render method. It supports:            
            - mode="rgb_array": Return a single frame representing the current state
            of the environment. A frame is a np.ndarray with shape (x, y, 3) .
        """
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
        self._send_command_to_process_wrapper(IPCCommands.rewind_state(self.ipc_cmd_id, reset_state))


class ContinuousTMNF_Single_Agent_Env(TMNF_Single_Agent_Env):
    def __init__(self, command_queue, response_queue, obs_manager, reward_calculator, termination_manger, 
                 track, reset_mode, n_previous_actions, position_buffer_size, position_moved_threshold, 
                 ignore_stuck_for_n_steps_after_reset, game_speed, countdown_speed, waitforstep_timeout_in_s, 
                 startposition_accuracy_threshold, gamma, actiondim : int, **kwargs):
        
        super().__init__(command_queue, response_queue, obs_manager, reward_calculator, termination_manger, 
                         track, reset_mode, n_previous_actions, position_buffer_size, position_moved_threshold,
                          ignore_stuck_for_n_steps_after_reset, game_speed, countdown_speed, waitforstep_timeout_in_s, 
                          startposition_accuracy_threshold, gamma, is_discrete=False, **kwargs)
        
        self.action_space = gym.spaces.Box(low=-1, high = 1, shape = (actiondim,), dtype=np.float32)
        self.actionmode = ActionMode.get_mode(continuous=True, dim = actiondim)
        self._send_command_to_process_wrapper(IPCCommands.set_actionmode(self.ipc_cmd_id, self.actionmode))

    def _get_action(self, action):
        # TODO this should also be done via the ActionMode class in the parent-environemnt
        return action