

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


class NextPointObsManager(ObservationManager):
    def __init__(self, observation_list, colorspace, convert_torch, img_width, img_height):
        super().__init__(observation_list, colorspace, convert_torch, img_width, img_height)
        self.reference_line_points_lookahead = 40
        self.refeence_line_stride = 10
        assert self.reference_line_points_lookahead % self.refeence_line_stride == 0
        self.last_obs : SimStateData = None

        self.statevector_dim = 0
        
    def get_observation_dict(self) -> spaces.Dict:
        """Returns observation dict for environment according to initialization."""
        self.statevector_dim = 5 + 3 * self.reference_line_points_lookahead \
            + 4*NUM_SURFACE_CATEGORIES + 3*4 + 4*1 # get_mobile_states


        return spaces.Dict({
                "image": spaces.Box(low=0, high=1.0, shape=(self.n_channels, self.img_height, self.img_width), dtype=np.uint8),
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(self.statevector_dim,), dtype=np.float32),
            })
    

    def get_values_from_state_dict(self, obs : SimStateData):

        dyna_current: HmsDynaStateStruct = obs.dyna.current_state

        mobile_states = self.get_mobil_states(obs)
        next_idx, d, drel = self.env.reference_line.get_distance_to_next_point()
        lateral_dist = self.env.reference_line.calculate_lateral_difference(next_idx, obs.position)

        velocity_delta : np.ndarray =  np.array([0,0,0])
        if not self.last_obs == None:
            velocity_delta = np.array(obs.velocity) - np.array(self.last_obs.velocity)

        
        orientation = np.array(dyna_current.rotation.to_numpy(), dtype=float).T
        comming_refline_points = self.get_next_refline_points(next_idx, obs.position, orientation)


        self.last_obs = obs

        floatvec : np.ndarray = np.hstack([drel,
                                            velocity_delta,
                                            lateral_dist,
                                            comming_refline_points.ravel(),
                                            mobile_states], dtype =  np.float32)
        
        assert floatvec.shape[0] == self.statevector_dim, f"Floatvector has size {floatvec.shape[0]}, however should be size {self.statevector_dim}"
        
        self.info["position"] = obs.position
        self.info["orientation"] = orientation
        self.info["d"] = d
        self.info["drel"] = drel
        self.info["velocity_delta"] = velocity_delta
        self.info["comming_refline_points"] = comming_refline_points
        self.info["mobile_states"] = mobile_states
        self.info["lateral_dist"] = lateral_dist

        return floatvec


    def get_next_refline_points(self, next_refline_idx : int, car_position : np.ndarray, car_orientation : np.ndarray) -> np.ndarray:
        comming_refline_points = self.env.reference_line.get_reference_line_points(begin_idx=next_refline_idx,
                                                                end_idx= next_refline_idx + self.reference_line_points_lookahead * self.refeence_line_stride, 
                                                                extrapolate= True, 
                                                                stride = self.refeence_line_stride)
        
        assert comming_refline_points.shape[0] == self.reference_line_points_lookahead, f"Expected to get {self.reference_line_points_lookahead} points from reflinemanager, got {comming_refline_points.shape[0]}"
        
        comming_refline_points_rel_to_car : np.ndarray = np.array(car_orientation).dot((comming_refline_points - np.array(car_position)).T).T # (n,3)
        #comming_refline_points_rel_to_car_flattened = comming_refline_points_rel_to_car.ravel()
        return comming_refline_points_rel_to_car


    def get_mobil_states(self, game_states:SimStateData):
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
