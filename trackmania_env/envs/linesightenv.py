"""
Custom Gymnasium Environment 
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
"""
from typing import Any,Dict,Tuple,Optional,List

import gymnasium as gym
from gymnasium.spaces import Box,Discrete 
from numpy import uint8,int32,float32,inf
import numpy as np
import numpy.typing as npt
import time 
from queue import Queue
from tminterface.structs import (
    CheckpointData, 
    SimStateData, 
    CheckpointTime,
    HmsDynaStateStruct,
    HmsDynaStruct,
    SceneVehicleCar,
    SimulationWheel,
    RealTimeState,
    Engine)

from game_interaction.process_wrapper import TMIProcessWrapper
from game_interaction.tminterface_commands import TMInterfaceCommands

from trackmania_env.utils.position_buffer import PositionBuffer
from trackmania_env.utils.actionmap import ACTION_MAP
from trackmania_env.utils.contact_materials import physics_behavior_fromint,NUM_SURFACE_CATEGORIES
from trackmania_env.envs.rewards.reward_calculation import RewradCalculator

from simstate_space_dict import simstate_space_dict

import re
import logging
import functools

from bytefield import ByteArrayField,IntegerField,FloatField,BooleanField
import torch

from collections import deque

class Linesight_Single_Agent_Env(gym.Env):
    """The reinforcement learning environment for Trackmania Nations Forever"""

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}
    
    # init, step and reset have to be implemented for the class to be gym compatible
    def __init__(
            self,
            command_queue: Queue,
            response_queue: Queue,
            zone_centers: np.ndarray,
            zone_transitions: np.ndarray,
            distance_between_zone_transitions: np.ndarray,
            distance_from_start_track_to_prev_zone_transition: np.ndarray,
            normalized_vector_along_track_axis: np.ndarray,
            next_real_checkpoint_positions: np.ndarray,
            max_allowable_distance_to_real_checkpoint: np.ndarray,
            cfg,
            position_buffer_size: int = 20,
            position_moved_threshold: float = 0.2,
            reset_mode: str = "respawn",
            reward_calculator: str = "basic",
            n_prev_actions:int = 10,
            ):
        """
        Initializes the custom Gymnasium environment.
        This constructor sets up the basic structure of the environment.
        As required by Gymnasium environments, it defines the action and observation spaces.
        We also define some other important varibales in order to communicate with TMInterface
        """ 
        self.cfg = cfg # config file for this environment
        self.zone_centers: npt.NDArray = zone_centers
        self.current_zone_idx:int = cfg.n_zone_centers_extrapolate_after_end_of_map
        self.distance_since_track_begin:int = 0
        self.max_allowable_distance_to_virtual_checkpoint = np.sqrt((cfg.distance_between_checkpoints / 2) ** 2 + (cfg.road_width / 2) ** 2)

        self.zone_transitions = zone_transitions
        self.distance_between_zone_transitions = distance_between_zone_transitions
        self.distance_from_start_track_to_prev_zone_transition = distance_from_start_track_to_prev_zone_transition
        self.normalized_vector_along_track_axis = normalized_vector_along_track_axis
        self.next_real_checkpoint_positions = next_real_checkpoint_positions
        self.max_allowable_distance_to_real_checkpoint = max_allowable_distance_to_real_checkpoint
        self.n_prev_actions = n_prev_actions

        self.float_input_dim =  (
            # dynamic states sizes (see get_dynamics_states() for understanding)
            3* self.cfg.n_zone_centers_in_inputs 
            + 3*3 
            #--------------------------
            # wheels and engine states sizes (see get_mobil_states() for understanding)
            + 4*NUM_SURFACE_CATEGORIES
            + 3*4 + 4*1
            #--------------------------
            # previous actions
            + 4* n_prev_actions
            +1 # min_dist
        )

        self.observation_space = gym.spaces.Dict(
            {
                "image" : gym.spaces.Box(low=0, high=255, shape=(100,100,4), dtype=np.uint8),
                "floats" : gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.float_input_dim,), dtype=np.float32),
            }
        )
        self.observations = {}
        self.action_space = gym.spaces.Discrete(len(ACTION_MAP))
        #self.actions:deque = deque(maxlen=n_prev_actions)
        self.actions:deque = deque([(False,False,False,False)] * self.n_prev_actions, maxlen=self.n_prev_actions)
        """list of actions that may be stored later."""

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

        self.rew_calculator = RewradCalculator.get_instance(reward_calculator, self.position_buffer)
        
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

        race_time:int = game_states.race_time

        # =========================
        # Dynamic Car State
        # =========================
        (
        state_zone_center_coordinates_in_car_reference_system,
        y_map_vector_in_car_reference_system,
        velocity_in_car_reference_system,
        angular_velocity_in_car_reference_system
        ) = self.get_dynamics_states(game_states=game_states)

        # =======================================
        # Wheel States,Engine and Gearbox State
        # =======================================
        car_gear_and_wheels = self.get_mobil_states(game_states=game_states)

        # put all state information into a combined vector
        floats = np.hstack(
                        (
                            #0,# TODO what is this 0 for ?
                            # also pass the previous actions as input to the NN. 
                            # from Linesight: 
                            #   pb4's theory was that it would help understand neoslides where you have to steer in a direction, 
                            #   not steer then steer and brake. We are not sure if it is really necessary to have these inputs
                            np.array(self.actions).ravel(),
                            car_gear_and_wheels.ravel(),
                            angular_velocity_in_car_reference_system.ravel(),
                            velocity_in_car_reference_system.ravel(),
                            y_map_vector_in_car_reference_system.ravel(),
                            state_zone_center_coordinates_in_car_reference_system.ravel(),
                            min(
                                self.cfg.margin_to_announce_finish_meters,
                                self.distance_from_start_track_to_prev_zone_transition[
                                    len(self.zone_centers) - self.cfg.n_zone_centers_extrapolate_after_end_of_map
                                ]
                                - self.distance_since_track_begin,
                            ),
                        )
                    ).astype(np.float32)
        sim_step : int = imgs_and_simstate["sim_step"]

        self.observations["image"] = image
        self.observations["floats"] = floats

        return (self.observations,game_states)

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

        observations,ssD = self._get_obs()
        self.position_buffer.add(ssD.position)
        race_finished = ssD.player_info.race_finished
    
        # TODO check if episode terminated and or tuncated, terminated if reached the goal, truncated means that a timelimit has been reached but MDP is not in a terminal state
        # TODO look at the target and current approach in linesight_rl gim line 600-601 or maybe just stick with race_is_finished
        stuck = not self.position_buffer.moved_more_than_threshold(self.position_buffer_threshold)
        terminated = not self.position_buffer.moved_more_than_threshold(self.position_buffer_threshold) or race_finished
        truncated = False
        
        reward = self.rew_calculator.calculate_reward(ssD, race_finished, stuck)
        
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
        observation,ssD = self._get_obs()
        info = self._get_info()
        # can not clear since later shapes get uncompatible
        self.actions = deque([(False,False,False,False)] * self.n_prev_actions, maxlen=self.n_prev_actions)
        self.current_zone_idx = self.cfg.n_zone_centers_extrapolate_after_end_of_map
        self.distance_since_track_begin = 0
        self.reset_car(ssD.position)
        self.position_buffer.reset()
        self.rew_calculator.reset()

        return observation, info
    
    def get_dynamics_states(self,game_states: SimStateData):
        # =========================
        # Dynamic Car State
        # =========================
        # Represents the current dynamic state of the car, such as its position, orientation, speed ... .
        dyna_current: HmsDynaStateStruct = game_states.dyna.current_state
        position = np.array(dyna_current.position,dtype=np.float32,)  # (3,)
        orientation = dyna_current.rotation.to_numpy().T  # (3, 3)
        velocity = np.array(dyna_current.linear_speed,dtype=np.float32)  # (3,)
        angular_speed = np.array(dyna_current.angular_speed,dtype=np.float32)  # (3,)

        # Compute how far along the agent is within the current zone
        meters_in_current_zone = np.clip(
            (position - self.zone_transitions[self.current_zone_idx - 1]).dot(
                self.normalized_vector_along_track_axis[self.current_zone_idx - 1]
            ),
            0,
            self.distance_between_zone_transitions[self.current_zone_idx - 1],
        )

        # Total distance from the start of the track
        self.distance_since_track_begin = (
            self.distance_from_start_track_to_prev_zone_transition[self.current_zone_idx - 1]
            + meters_in_current_zone
        )
        
        # TODO does this have to be here or only in the step function? 
        # deck height is set to -np.inf 
        if position[1] >  -np.inf: # change this to be not hard coded 
                        self.current_zone_idx = self.update_current_zone_idx(
                            self.current_zone_idx,
                            self.zone_centers,
                            position,
                            self.max_allowable_distance_to_virtual_checkpoint,
                            self.next_real_checkpoint_positions,
                            self.max_allowable_distance_to_real_checkpoint,
                        )
        """
        converting global/world-frame vectors into the car's local reference frame
        which should help the neural network of the agent to learn better since 
        inputs are then consistent in scale, orientation, ... -> agent (car) needs to understand 
        how it is moving relative to itself, not the world -> lets the policy generalize (in theory)
        From tminterface discord :
            - X: points to the left
            - Y: points upwards
            - Z: points forwards
        """
        state_zone_center_coordinates_in_car_reference_system = orientation.dot(
            (
                self.zone_centers[
                    # Get a slice of zone centers starting from `current_zone_idx`, spaced by `one_every_n_zone_centers_in_inputs`
                    # We are selecting `n_zone_centers_in_inputs` entries in total.
                    # -------------------------
                    # Example With Real Numbers
                    # -------------------------
                    # one_every_n_zone_centers_in_inputs = 20
                    # n_zone_centers_in_inputs = 40
                    # current_zone_idx = 100
                    # The slicing becomes:
                    # self.zone_centers[100 : 100 + 800 : 20, :] → self.zone_centers[100:900:20, :]
                    # This picks the following indices: [100, 120, 140, ..., 880] → total of 40 zone centers
                    self.current_zone_idx : self.current_zone_idx + self.cfg.one_every_n_zone_centers_in_inputs
                                * self.cfg.n_zone_centers_in_inputs : self.cfg.one_every_n_zone_centers_in_inputs,
                                :,
                            ]
                            - position
                        ).T
                    ).T  # (n_zone_centers_in_inputs, 3)
        y_map_vector_in_car_reference_system = orientation.dot(np.array([0, 1, 0])) #(3,)
        velocity_in_car_reference_system = orientation.dot(velocity)  #(3,)
        angular_velocity_in_car_reference_system = orientation.dot(angular_speed)  #(3,)
        return (
             state_zone_center_coordinates_in_car_reference_system,
             y_map_vector_in_car_reference_system,
             velocity_in_car_reference_system,
             angular_velocity_in_car_reference_system)
    
    def get_mobil_states(self,game_states:SimStateData):
        # =======================================
        # Wheel States,Engine and Gearbox State
        # =======================================
        # gather all relevant information which describe the engine and the wheels states and pack them into a single array
        mobil:SceneVehicleCar = game_states.scene_mobil
        engine:Engine = mobil.engine
        gearbox_state = mobil.gearbox_state

        wheels: np.ndarray[SimulationWheel] = game_states.simulation_wheels
        wheels_states: List[RealTimeState] = [wheels[i].real_time_state for i in range(wheels.shape[0])]

        car_gear_and_wheels = np.array(
            [
                *(ws.is_sliding for ws in wheels_states),  # Bool (* is for unpacking)              size: 4
                *(ws.has_ground_contact for ws in wheels_states),  # Bool                           size: 4
                *(ws.damper_absorb for ws in wheels_states),  # 0.005 min, 0.15 max, 0.01 typically size: 4
                gearbox_state,  # Bool, except 2 at startup                                         size: 1
                engine.gear,  # 0 -> 5 approx                                                       size: 1
                engine.actual_rpm,  # 0-10000 approx                                                size: 1
                mobil.is_freewheeling, # Bool                                                       size: 1
                # linesight (from tminterface discord):
                #    n_contact_material_physics_behavior_types (here NUM_SURFACE_CATEGORIES) is the number of possible 
                #    materials a wheel can touch, we added this thinking it would help the agent understand the different
                #    behaviors of the car on different surfaces by having the info more than visually, again we are not sure 
                #    if this input is useful we did not do proper ablation tests because each test is very long.
                # TODO: Could we only store 4 values saying on which surface each wheel is instead of 4*NUM_SURFACE_CATEGORIES values
                *(
                    i == physics_behavior_fromint[ws.contact_material_id & 0xFFFF] # Doing & 0xFFFF masks the value, keeping only the lowest 16 bits and discarding any higher bits.
                    for ws in wheels_states
                    for i in range(NUM_SURFACE_CATEGORIES)
                ),                                                                              #    size: 4*NUM_SURFACE_CATEGORIES
            ],dtype=np.float32,)
        return car_gear_and_wheels

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

    

#@numba.njit
    def update_current_zone_idx(self,
        current_zone_idx: int,
        zone_centers: npt.NDArray,
        sim_state_position: npt.NDArray,
        max_allowable_distance_to_virtual_checkpoint: float,
        next_real_checkpoint_positions: npt.NDArray,
        max_allowable_distance_to_real_checkpoint: npt.NDArray,):
        d1 = np.linalg.norm(zone_centers[current_zone_idx + 1] - sim_state_position)
        d2 = np.linalg.norm(zone_centers[current_zone_idx] - sim_state_position)
        d3 = np.linalg.norm(zone_centers[current_zone_idx - 1] - sim_state_position)
        d4 = np.linalg.norm(next_real_checkpoint_positions[current_zone_idx] - sim_state_position)
        while (
            d1 <= d2
            and d1 <= max_allowable_distance_to_virtual_checkpoint
            and current_zone_idx
            < len(zone_centers) - 1 - self.cfg.n_zone_centers_extrapolate_after_end_of_map  # We can never enter the final virtual zone
            and d4 < max_allowable_distance_to_real_checkpoint[current_zone_idx] ):
            # Move from one virtual zone to another
            current_zone_idx += 1
            d2, d3 = d1, d2
            d1 = np.linalg.norm(zone_centers[current_zone_idx + 1] - sim_state_position)
            d4 = np.linalg.norm(next_real_checkpoint_positions[current_zone_idx] - sim_state_position)
        while current_zone_idx >= 2 and d3 < d2 and d3 <= max_allowable_distance_to_virtual_checkpoint:
            current_zone_idx -= 1
            d1, d2 = d2, d3
            d3 = np.linalg.norm(zone_centers[current_zone_idx - 1] - sim_state_position)
        
        return current_zone_idx