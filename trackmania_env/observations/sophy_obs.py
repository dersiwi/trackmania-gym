import numpy as np
from collections import deque
from gymnasium import spaces
import torch
from trackmania_env.observations.observation_manager import ObservationManager
from game_interaction.ipc_fields import IPCFields
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

IMAGE_SIZE = 64
MS_TO_KMH = 3.6       # meters per second to kilometers per hour
MILLISECONDS_TO_SECONDS = 1000  # milliseconds to seconds

class SophyObsManager(ObservationManager):
    def __init__(self, observation_list, colorspace, convert_torch, img_width, img_height,maxlen_history:int = 3):
        assert img_width == img_height == IMAGE_SIZE, (
            f"Sophy was trained on {IMAGE_SIZE}x{IMAGE_SIZE} images. "
            "Please use square images of this size."
        )
        super().__init__(observation_list, colorspace, convert_torch, img_width, img_height)
        self.last_velocity = np.array([0.,0.,0.],dtype=np.float32)
        self.maxlen_history = maxlen_history
        self.angles : deque = deque([0.]*self.maxlen_history, maxlen=self.maxlen_history) # History of the last three steering angles
        self.last_time = 0 # dunno if this should be set to zero after env reset because in game timer doesn't reset 
        self.epsilon = 1e-6  # Tolerance for float comparisons
        self.max_degree_radians = np.pi / 6 # equivalent to np.deg2rad(30) so the max steering angle is |30| degree 

        self.propriocentric_features_dim = 3*4 + 2*self.maxlen_history - 1
        self.global_features_dim = 60 * 3

    def reset(self):
        self.last_velocity = np.array([0.,0.,0.],dtype=np.float)
        self.h_a_t = np.array([0.,0.,0.],dtype=np.float)
        self.angles : deque = deque([0.]*self.maxlen_history, maxlen=self.maxlen_history)

    def get_observation_dict(self):

        return spaces.Dict({
                "image": spaces.Box(low=0, high=1.0, shape=(self.n_channels, self.img_height, self.img_width), dtype=np.uint8),
                "propriocentric_features": spaces.Box(low=-np.inf, high=np.inf, shape=(self.propriocentric_features_dim,), dtype=np.float32),
                "global_features": spaces.Box(low=-np.inf, high=np.inf, shape=(self.global_features_dim,), dtype=np.float32),
            })
    
    def get_observation(self, raw_observation : dict[str, np.ndarray | SimStateData]) -> tuple[np.ndarray | dict[str, np.ndarray] | torch.Tensor | dict[str, torch.Tensor],dict[str,any]]:
        """
        Takes raw observations from TMInterface and dissects them into image
        """
        game_states = raw_observation[IPCFields.SIMSTATE]
        propriocentric_features = self.get_propriocentric_features(game_states)
        global_features = self.get_global_features(game_states)
        img = self.cnvt_imgs(raw_observation[IPCFields.IMG])
        
        assert img.shape == (self.n_channels, self.img_height, self.img_width), f"Expected shape to be ({self.n_channels},{self.img_height}, {self.img_width}) but got {img.shape}"

        propriocentric_features = torch.from_numpy(propriocentric_features).float()
        global_features = torch.from_numpy(global_features).float() 

        return {"image": img, "propriocentric_features": propriocentric_features, "global_features": global_features}, self.info
    
    def cnvt_imgs(self, images):
        assert self.colorspace == ObservationManager.Colorspace.RGB, (
        "Invalid colorspace: Sophy was trained using RGB images, as described in the original paper. "
        "Please use RGB colorspace."
        )
        return super().cnvt_imgs(images)
    
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
        position = np.array(dyna_current.position,dtype=np.float32,)  # (3,)
        orientation = dyna_current.rotation.to_numpy().T.astype(np.float32)  # (3, 3)
        velocity = np.array(dyna_current.linear_speed,dtype=np.float32)  # (3,)
        angular_speed = np.array(dyna_current.angular_speed,dtype=np.float32)  # (3,)
        time =  game_states.time/MILLISECONDS_TO_SECONDS
        delta_t = time - self.last_time # normal time is counted in ms
        assert delta_t != 0

        v_t:np.ndarray = orientation.dot(velocity)  #(3,)
        a_t:np.ndarray =(v_t - self.last_velocity)/delta_t #(3,)
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
 
    def get_global_features(self, game_states: SimStateData):
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
        #NOTE as for now we only return the center points no left and right
        # and will only work with the sophy reference line 

        dyna_current: HmsDynaStateStruct = game_states.dyna.current_state # current dynamic state of the car, such as its position, orientation, speed ... .
        position = np.array(dyna_current.position,dtype=np.float32,)  # (3,)
        orientation = dyna_current.rotation.to_numpy().T.astype(np.float32)  # (3, 3)
        speed = game_states.display_speed / MS_TO_KMH

        next_idx, d, drel = self.env.reference_line.get_distance_to_next_point()
        comming_refline_points = self.env.reference_line.get_reference_line_points(begin_idx= next_idx ,speed = speed ,extrapolate = True)
        comming_refline_points : np.ndarray = np.array(orientation).dot((comming_refline_points - np.array(position)).T).T

        self.info["comming_refline_points"] = comming_refline_points
        self.info["orientation"] = orientation
        self.info["position"] = position
     
        return comming_refline_points.ravel()
