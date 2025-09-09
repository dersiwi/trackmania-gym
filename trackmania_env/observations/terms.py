import numpy as np

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


from trackmania_env.utils.contact_materials import physics_behavior_fromint, NUM_SURFACE_CATEGORIES
from trackmania_env.utils.constants import ObsNormalizationFactors
from trackmania_env.utils.reference_line_manager import ReferenceLineManager
from trackmania_env.utils.interpolation import interpolate_points
import trackmania_env.utils.constants as constants
from collections import deque

from utils.image_converter import ImageConverter

def get_wheel_states(game_states : SimStateData) -> tuple[list[RealTimeState], np.ndarray[SimulationWheel]]:
    wheels: np.ndarray[SimulationWheel] = game_states.simulation_wheels
    wheels_states : list[RealTimeState] = [wheels[i].real_time_state for i in range(wheels.shape[0])]
    return wheels_states, wheels

from trackmania_env.observations.observation_manager import ObservationTerm
from game_interaction.ipc_fields import IPCFields


class SurfaceFloats(ObservationTerm):
    

    def __init__(self, normalize):
        super().__init__((4), normalize)

    def _normalize(self, obs):
        return obs # is always normalized

    def _get_obs(self, game_states : SimStateData):
        """# linesight (from tminterface discord): TODO : Big refactor!!
            #    n_contact_material_physics_behavior_types (here NUM_SURFACE_CATEGORIES) is the number of possible 
            #    materials a wheel can touch, we added this thinking it would help the agent understand the different
            #    behaviors of the car on different surfaces by having the info more than visually, again we are not sure 
            #    if this input is useful we did not do proper ablation tests because each test is very long.
            # 
        This method just transforms this one-hot-encoded vector [1. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0.] into [0.2 0.2 0.2 0.2]"""
        
        wheels_states, wheels = get_wheel_states(game_states)
        wsm = []
        for ws in wheels_states:
            # Doing & 0xFFFF masks the value, keeping only the lowest 16 bits and discarding any higher bits.
            scidx = np.nonzero([i == physics_behavior_fromint[ws.contact_material_id & 0xFFFF] for i in range(NUM_SURFACE_CATEGORIES)])[0]
            if len(scidx) == 0: wsm.append(0)
            else:
                assert len(scidx)  == 1, scidx
                wsm.append(scidx[0] + 1 / (NUM_SURFACE_CATEGORIES + 1)) # +1 because sometimes its just no surface category (wheel is in the air.)
        return np.array(wsm, dtype = np.float32)

class MobileStatesTerm(ObservationTerm):
    """Returns all relevant dynamic car states"""

    def __init__(self, normalize):
        super().__init__((16), normalize)

    def _normalize(self, obs):
        obs[12]             /= ObsNormalizationFactors.gearbox_norm
        obs[13]             /= ObsNormalizationFactors.gear_norm
        obs[14]             /= ObsNormalizationFactors.rpm_norm
        return obs
    
    def _get_obs(self, game_states : SimStateData) -> np.ndarray:
        """Wheel States,Engine and Gearbox State
        gather all relevant information which describe the engine and the wheels states and pack them into a single array"""
        mobil:SceneVehicleCar = game_states.scene_mobil
        engine:Engine = mobil.engine
        gearbox_state = mobil.gearbox_state

        wheels: np.ndarray[SimulationWheel] = game_states.simulation_wheels
        wheels_states: list[RealTimeState] = [wheels[i].real_time_state for i in range(wheels.shape[0])]

        car_gear_and_wheels = np.array(
            [
                *(ws.is_sliding for ws in wheels_states),  # Bool (* is for unpacking)              size: 4 0:4
                *(ws.has_ground_contact for ws in wheels_states),  # Bool                           size: 4 4:8
                *(ws.damper_absorb for ws in wheels_states),  # 0.005 min, 0.15 max, 0.01 typically size: 4 8:12
                gearbox_state,  # Bool, except 2 at startup                                         size: 1 12
                engine.gear,  # 0 -> 5 approx                                                       size: 1 13
                engine.actual_rpm,  # 0-10000 approx                                                size: 1 14
                mobil.is_freewheeling, # Bool                                                       size: 1 15
            ],dtype=np.float32,)

        return car_gear_and_wheels

class SpeedTerm(ObservationTerm):
    def __init__(self, normalize):
        super().__init__((1), normalize)

    def _normalize(self, obs):
        return obs / ObsNormalizationFactors.speed_norm

    def _get_obs(self, raw_observation : dict[str, np.ndarray | SimStateData], **kwargs):
        game_states : SimStateData = raw_observation[IPCFields.SIMSTATE]

        return game_states.display_speed

class NextReflinePoint(ObservationTerm):
    def __init__(self, n_refline_points : int, reference_line_stride : int, normalize):
        super().__init__((n_refline_points * 3), normalize)
        self.n_refline_points = n_refline_points
        self.reference_line_stride : int = reference_line_stride

    def _normalize(self, obs):
        return obs / ObsNormalizationFactors.refline_norm
    
    def _get_obs(self, raw_observation : dict[str, np.ndarray | SimStateData], **kwargs):
        game_states : SimStateData = raw_observation[IPCFields.SIMSTATE]
        dyna_current: HmsDynaStateStruct = game_states.dyna.current_state
        refline : ReferenceLineManager = self.env.reference_line
        next_idx, _, _ = refline.get_distance_to_next_point()
        comming_refline_points : np.ndarray = refline.get_reference_line_points(begin_idx=next_idx,
                                                                end_idx= next_idx + self.n_refline_points * self.reference_line_stride, 
                                                                extrapolate= True, stride = self.reference_line_stride)

        car_position = game_states.position
        car_orientation = np.array(dyna_current.rotation.to_numpy(), dtype=float).T

        assert comming_refline_points.shape[0] == self.n_refline_points, f"Expected to get {self.n_refline_points} points from reflinemanager, got {comming_refline_points.shape[0]}"

        comming_refline_points_rel_to_car : np.ndarray = np.array(car_orientation).dot((comming_refline_points - np.array(car_position)).T).T # (n,3)
        #comming_refline_points_rel_to_car_flattened = comming_refline_points_rel_to_car.ravel()
        return comming_refline_points_rel_to_car.ravel()
    
class LateralDistance(ObservationTerm):

    def __init__(self, normalize):
        super().__init__((1), normalize)
    
    def _normalize(self, obs):
        return obs / ObsNormalizationFactors.lateral_dist_norm

    def _get_obs(self, raw_observation : dict[str, np.ndarray | SimStateData], **kwargs):
        game_states : SimStateData = raw_observation[IPCFields.SIMSTATE]
        reference_line = self.env.reference_line
        next_idx, d, drel = reference_line.get_distance_to_next_point()
        lateral_distance : np.ndarray = reference_line.calculate_lateral_difference(next_idx, game_states.position)
        return lateral_distance
    
class RelativeDistance(ObservationTerm):
    def __init__(self, normalize):
        super().__init__((1), normalize)

    def _normalize(self, obs):
        return obs
    
    def _get_obs(self, raw_observation : dict[str, np.ndarray | SimStateData], **kwargs):
        reference_line = self.env.reference_line
        next_idx, d, drel = reference_line.get_distance_to_next_point()
        return drel

class SophyGlobalFeatures(ObservationTerm):

    def __init__(self, lookahead_sec, n_points : int, normalize):
        super().__init__((n_points * 3), normalize)
        self.lookahead_sec = lookahead_sec
        self.n_points = n_points

    def _get_obs(self, raw_observation : dict[str, np.ndarray | SimStateData], **kwargs):
        """
        Extracts global course point features from the input game states, following the method
        described in Wurman et al. (2022).
    
        Course points describe the geometry of the track via the left and right boundaries and the
        center line. At each time step, this method computes the 3D relative positions of course 
        points ahead of the agent, based on its current velocity. These points are sampled from 
        0.1 to 6.0 seconds into the future, at 0.1-second intervals, assuming constant forward speed.
    
        The distance to each course point is dynamically computed using the agent's current speed 
        (i.e., distance = velocity x time). This results in 59 course points per line (left, center, right), 
        giving a predictive spatial representation of the upcoming track segment.
    
        Args:
            game_states (SimStateData): The current simulation state(s), including vehicle speed and 
                position.
    
        Returns:
            np.ndarray or torch.Tensor: A tensor containing the relative 3D coordinates of the course 
                points for the left, center, and right track lines.
        """
        game_states : SimStateData = raw_observation[IPCFields.SIMSTATE]
        #NOTE as for now we only return the center points no left and right
        reference_line = self.env.reference_line
        dyna_current: HmsDynaStateStruct = game_states.dyna.current_state # current dynamic state of the car, such as its position, orientation, speed ... .
        
        position = np.array(dyna_current.position,dtype=np.float32,)  # (3,)
        orientation = dyna_current.rotation.to_numpy().T.astype(np.float32)  # (3, 3)
        speed = game_states.display_speed / constants.MS_TO_KMH

        next_idx, d, drel = reference_line.get_distance_to_next_point()
        if d > constants.MAX_DISTANCE_TO_REFLINE : speed = 0

        cp_passed = max(self.lookahead_sec * speed,1) // reference_line.mean_segment_length
        end_idx = int(next_idx + cp_passed)

        points =  reference_line.get_reference_line_points(
                    begin_idx= next_idx, 
                    end_idx= end_idx, 
                    extrapolate=True,
                    stride= 1)
        
        assert points.shape[0] != 0
        comming_refline_points = np.repeat(points, self.n_points, axis=0) if points.shape[0] == 1 else interpolate_points(n_points= self.n_points,points=points)
        comming_refline_points : np.ndarray = np.array(orientation).dot((comming_refline_points - np.array(position)).T).T

        self.info["comming_refline_points"] = comming_refline_points
        self.info["orientation"] = orientation
        self.info["position"] = position
     
        return comming_refline_points.ravel()
    

class ImageObservationTerm(ObservationTerm):
    """"""
    class Colorspace:
        GRAYSCALE = 0
        RGB = 1
        BGRA = 2

        REV_DICT = {"grayscale" : 0, "rgb" : 1, "rgba" : 2} #this is somewhat ugly but this way the config contains a readable string
    def __init__(self, width, height, normalize, colorspace : str, cvr_grayscale_to_uint8 : bool):
        self.n_channels = 1 if self.colorspace == ImageObservationTerm.Colorspace.GRAYSCALE else 3
        super().__init__((self.n_channels, height, width), normalize)
        self.colorspace = colorspace
        self.convert_grayscale_to_uint8 = cvr_grayscale_to_uint8

    def _get_obs(self, images : np.ndarray) -> np.ndarray:
        """Converts image given by simulation into specified colortype and normalizes them into [0,1]."""
        if self.colorspace == ImageObservationTerm.Colorspace.RGB:
            imgs = ImageConverter.bgra_to_rgb(images)
        elif self.colorspace == ImageObservationTerm.Colorspace.GRAYSCALE:
            imgs = ImageConverter.bgra_to_graysacle(images, self.convert_grayscale_to_uint8)

        # only normlalize if not conversion, otherwiese it'll be stored as float again.
        if self.colorspace == ImageObservationTerm.Colorspace.GRAYSCALE and self.convert_grayscale_to_uint8:
            return imgs
        else:
            return imgs / 255.0
        
class Propriocentric_features(ObservationTerm):
        
        def __init__(self, normalize, maxlen_history = 3, lookahead_sec = 6,n_points = 60):
            self.last_velocity = np.array([0.,0.,0.],dtype=np.float32)
            self.maxlen_history = maxlen_history
            self.angles : deque = deque([0.]*self.maxlen_history, maxlen=self.maxlen_history) # History of the last three steering angles
            self.last_time = 0 # dunno if this should be set to zero after env reset because in game timer doesn't reset 
            self.epsilon = 1e-6  # Tolerance for float comparisons
            self.max_degree_radians = np.pi / 6 # equivalent to np.deg2rad(30) so the max steering angle is |30| degree 
            
            self.lookahead_sec = lookahead_sec
            self.n_points = n_points
            self.propriocentric_features_dim = 3*4 + 2*self.maxlen_history - 1
            super().__init__((self.propriocentric_features_dim), normalize) # TODO !! Fix shape


        def reset(self):
            self.last_velocity = np.array([0.,0.,0.],dtype=np.float32)
            self.h_a_t = np.array([0.,0.,0.],dtype=np.float32)
            self.angles : deque = deque([0.]*self.maxlen_history, maxlen=self.maxlen_history)
            
        def get_propriocentric_features(self, game_states: SimStateData):
            """
            Extracts propriocentric features from the given game state.

            Propriocentric features are derived from the car's local frame of reference 
            The extracted feature vector `opt` includes:

            - v_t (R^3): Linear velocity of the vehicle.
            - a_t (dv/dt) (R^3): Linear acceleration of the vehicle.
            - v_r_t (R^3): Angular velocity of the vehicle.
            - c_t (R^3): Control input vector consisting of steering, throttle, and brake values.
            - h_a_t (R^n): History of the last n (default is 3) steering angles.
            - h_d_t (R^n-1): History of delta (change in) steering values over the last n steps.

            Args:
                game_states (SimStateData): The current simulation state from which to extract 
                                            propriocentric features.

            Returns:
                np.ndarray: A concatenated array of all propriocentric features (opt) 
            """
            
            dyna_current: HmsDynaStateStruct = game_states.dyna.current_state # current dynamic state of the car, such as its position, orientation, speed ... .
            orientation = dyna_current.rotation.to_numpy().T.astype(np.float32)  # (3, 3)
            velocity = np.array(dyna_current.linear_speed,dtype=np.float32)  # (3,)
            angular_speed = np.array(dyna_current.angular_speed,dtype=np.float32)  # (3,)
            time =  game_states.time/constants.MILLISECONDS_TO_SECONDS
            delta_t = time - self.last_time # normal time is counted in ms
            #print(delta_t)
            #assert delta_t != 0

            v_t:np.ndarray = orientation.dot(velocity)  #(3,)
            a_t:np.ndarray =np.array([0.,0.,0.]) if delta_t == 0 else (v_t - self.last_velocity)/delta_t #(3,) # the only time delta_t should be zero is during the start
            v_r_t:np.ndarray = orientation.dot(angular_speed)  #(3,)

            throttle = 2.* game_states.scene_mobil.input_gas  - 1. # [-1,1] | original gas is between [0,1]
            brake =  2.* game_states.scene_mobil.input_brake -1. # [-1,1] | original brake is between [0,1]
            steer = game_states.scene_mobil.input_steer # [-1,1]
            c_t:np.ndarray = np.array([steer,throttle,brake]) #(3,)
            assert np.all(np.abs(c_t) <= 1 + self.epsilon), f"Values in c_t out of range [-1,1]: {c_t}"


            # steering angle
            turning_rate = game_states.scene_mobil.turning_rate # [-1,1]
            steering_angle = self.max_degree_radians * turning_rate # when wheels are fully to the right then turning rate is equal to 1 but they only rotate too roughly 30 degree.
            assert np.abs(steering_angle) <= self.max_degree_radians + self.epsilon, f"steering_angle out of range: {steering_angle}"
            self.angles.append(steering_angle)
            h_a_t:np.ndarray = np.array(self.angles)
            h_d_t:np.ndarray = h_a_t[1:] - h_a_t[:-1]

            
            self.last_time =  time
            self.last_velocity = v_t

            # for debuggin purposes
            self.info["linear_velocity"] = v_t
            self.info["linear_acceleration"] = a_t
            self.info["angular_velocity"] = v_r_t

            self.info["throttle"] = throttle
            self.info["brake"] = brake 
            self.info["steer"] = steer

            self.info["history_steering_angles"] = h_a_t
            self.info["history_steering_deltas"] = h_d_t

            propriocentric_features: np.ndarray = np.hstack([
                v_t.ravel(),    #(3,)
                a_t.ravel(),    #(3,)
                v_r_t.ravel(),  #(3,)
                c_t.ravel(),    #(3,)
                h_a_t.ravel(),  #(n,)
                h_d_t.ravel(),  #(n-1,)
            ],dtype =  np.float32) # total (3*4 + 2*n - 1)

            return propriocentric_features