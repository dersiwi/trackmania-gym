import numpy as np
from collections import deque
from trackmania_env.observations.observation_manager import ObservationManager
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

class SophyObsManager(ObservationManager):
    def __init__(self, observation_list, colorspace, convert_torch, img_width, img_height,maxlen_history:int = 3):
        super().__init__(observation_list, colorspace, convert_torch, img_width, img_height)
        self.last_velocity = np.array([0.,0.,0.],dtype=np.float)
        self.maxlen_history = maxlen_history
        self.actions : deque = deque([0.]*self.maxlen_history, maxlen=self.maxlen_history) # History of the last three steering angles
        self.last_time = 0 # dunno if this should be set to zero after env reset because in game timer doesn't reset 

    def reset(self):
        self.last_velocity = np.array([0.,0.,0.],dtype=np.float)
        self.h_a_t = np.array([0.,0.,0.],dtype=np.float)
        self.actions : deque = deque([0.]*self.maxlen_history, maxlen=self.maxlen_history)

    def get_observation_dict(self):
        return super().get_observation_dict()
    
    def get_values_from_state_dict(self, obs):
        return super().get_values_from_state_dict(obs)
    
    def get_propriocentric_features(self, game_states: SimStateData):
        """
        Extracts propriocentric features from the given game state.

        Propriocentric features are derived from the car's local frame of reference 
        The extracted feature vector `opt` includes:

        - v_t (R^3): Linear velocity of the vehicle.
        - a_t (dv/dt) (R^3): Linear acceleration of the vehicle.
        - v_r_t (R^3): Angular velocity of the vehicle.
        - c_t (R^3): Control input vector consisting of steering, throttle, and brake values.
        - h_a_t (R^3): History of the last three steering angles.
        - h_d_t (R^2): History of delta (change in) steering values over the last three steps.

        Args:
            game_states (SimStateData): The current simulation state from which to extract 
                                        propriocentric features.

        Returns:
            np.ndarray: A concatenated array of all propriocentric features (opt) in the order 
        """
        
        dyna_current: HmsDynaStateStruct = game_states.dyna.current_state # current dynamic state of the car, such as its position, orientation, speed ... .
        position = np.array(dyna_current.position,dtype=np.float32,)  # (3,)
        orientation = dyna_current.rotation.to_numpy().T  # (3, 3)
        velocity = np.array(dyna_current.linear_speed,dtype=np.float32)  # (3,)
        angular_speed = np.array(dyna_current.angular_speed,dtype=np.float32)  # (3,)
        delta_t = game_states.time - self.last_time
        assert delta_t != 0

        v_t = orientation.dot(velocity)  #(3,)
        a_t =(v_t - self.last_velocity)/delta_t #(3,)
        v_r_t = orientation.dot(angular_speed)  #(3,)

        throttle = 2.*game_states.input_gas  - 1. # [-1,1] | original gas is between [0,1]
        brake =  2.* game_states.input_brake -1. # [-1,1] | original brake is between [0,1]
        steer = game_states.input_steer # [-1,1]
        c_t = np.array([steer,throttle,brake]) #(3,)

        # steering angle
        

        self.last_time =  game_states.time
        self.last_velocity = v_t