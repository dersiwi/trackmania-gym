
from trackmania_env.observations.observation_manager import ObservationManager
from gymnasium import spaces
import numpy as np
from configs.config import LinesightObsCfg

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
from trackmania_env.utils.contact_materials import physics_behavior_fromint,NUM_SURFACE_CATEGORIES


class LinesightObservationWrapper(ObservationManager):

    def __init__(self, observation_list, colorspace, convert_torch, img_width, img_height,
            zone_centers: np.ndarray,
            zone_transitions: np.ndarray,
            distance_between_zone_transitions: np.ndarray,
            distance_from_start_track_to_prev_zone_transition: np.ndarray,
            normalized_vector_along_track_axis: np.ndarray,
            next_real_checkpoint_positions: np.ndarray,
            max_allowable_distance_to_real_checkpoint: np.ndarray,
            cfg : LinesightObsCfg):
        super().__init__(observation_list, colorspace, convert_torch, img_width, img_height)

        self.cfg = cfg # config file for this environment
        self.zone_centers: np.ndarray = zone_centers
        self.current_zone_idx:int = cfg.n_zone_centers_extrapolate_after_end_of_map
        self.distance_since_track_begin:int = 0
        self.state_zone_center_coordinates_in_car_reference_system : np.ndarray = np.zeros(3,)
        self.max_allowable_distance_to_virtual_checkpoint = np.sqrt((cfg.distance_between_checkpoints / 2) ** 2 + (cfg.road_width / 2) ** 2)

        self.zone_transitions = zone_transitions
        self.distance_between_zone_transitions = distance_between_zone_transitions
        self.distance_from_start_track_to_prev_zone_transition = distance_from_start_track_to_prev_zone_transition
        self.normalized_vector_along_track_axis = normalized_vector_along_track_axis
        self.next_real_checkpoint_positions = next_real_checkpoint_positions
        self.max_allowable_distance_to_real_checkpoint = max_allowable_distance_to_real_checkpoint

    def get_observation_dict(self) -> spaces.Dict:
        """Returns observation dict for environment according to initialization."""
        
        n_channels = 1 if self.colorspace == ObservationManager.Colorspace.GRAYSCALE else 3
        float_input_dim =  (
            # dynamic states sizes (see get_dynamics_states() for understanding)
            3* self.cfg.n_zone_centers_in_inputs 
            + 3*3 
            #--------------------------
            # wheels and engine states sizes (see get_mobil_states() for understanding)
            + 4*NUM_SURFACE_CATEGORIES
            + 3*4 + 4*1
            #--------------------------
            # previous actions
            + 4* self.env.n_prev_actions
            +1 # min_dist
        )
        return spaces.Dict({
                "image": spaces.Box(low=0, high=255, shape=(n_channels,self.img_width, self.img_height), dtype=np.uint8),
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(float_input_dim,), dtype=np.float32),
            })
    

    def get_values_from_state_dict(self, game_states : SimStateData):
        # =========================
        # Dynamic Car State
        # =========================
        (
        self.state_zone_center_coordinates_in_car_reference_system,
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
                            np.array(self.env.actions).ravel(),
                            car_gear_and_wheels.ravel(),
                            angular_velocity_in_car_reference_system.ravel(),
                            velocity_in_car_reference_system.ravel(),
                            y_map_vector_in_car_reference_system.ravel(),
                            self.state_zone_center_coordinates_in_car_reference_system.ravel(),
                            min(
                                self.cfg.margin_to_announce_finish_meters,
                                self.distance_from_start_track_to_prev_zone_transition[
                                    len(self.zone_centers) - self.cfg.n_zone_centers_extrapolate_after_end_of_map
                                ]
                                - self.distance_since_track_begin,
                            ),
                        )
                    ).astype(np.float32)
        
        return floats
    

    def get_mobil_states(self,game_states:SimStateData):
        # =======================================
        # Wheel States,Engine and Gearbox State
        # =======================================
        # gather all relevant information which describe the engine and the wheels states and pack them into a single array
        mobil:SceneVehicleCar = game_states.scene_mobil
        engine:Engine = mobil.engine
        gearbox_state = mobil.gearbox_state

        wheels: np.ndarray[SimulationWheel] = game_states.simulation_wheels
        wheels_states: list[RealTimeState] = [wheels[i].real_time_state for i in range(wheels.shape[0])]

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
    

    def update_current_zone_idx(self,
        current_zone_idx: int,
        zone_centers: np.ndarray,
        sim_state_position: np.ndarray,
        max_allowable_distance_to_virtual_checkpoint: float,
        next_real_checkpoint_positions: np.ndarray,
        max_allowable_distance_to_real_checkpoint: np.ndarray,):
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