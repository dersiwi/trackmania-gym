from collections import deque

import numpy as np
import gymnasium as gym

from trackmania_env.observations.observation_term import ObservationTerm
from trackmania_env.utils import constants
from game_interaction.ipc_fields import IPCFields
from tminterface.structs import SimStateData, HmsDynaStateStruct
from trackmania_env.utils.interpolation import interpolate_points

class PropriocentricTerm(ObservationTerm):
    # === Class-level constants ===
    MAX_LINEAR_SPEED = 1000.0          # Known upper bound of vehicle speed
    MAX_ACCEL = 50.0                   # Assumed reasonable bound for acceleration
    MAX_ANGULAR_SPEED = 20.0           # Assumed reasonable bound for angular velocity
    MAX_CONTROL = 1.0                  # Controls already in [-1, 1]
    MAX_STEER_ANGLE = np.pi / 6        # ~30 degrees in radians
    MAX_STEER_DELTA = 2 * np.pi / 6    # Assumed full swing delta between steps
    EPSILON = 1e-6                     # Tolerance for float comparisons

    def __init__(self, name = "propriocentric features", normalize=False,maxlen_history:int = 3):
        """
        Propriocentric features are derived from the car's local frame of reference 
        The extracted feature vector includes:

        - v_t (R^3): Linear velocity of the vehicle.
        - a_t (dv/dt) (R^3): Linear acceleration of the vehicle.
        - v_r_t (R^3): Angular velocity of the vehicle.
        - c_t (R^3): Control input vector consisting of steering, throttle, and brake values.
        - h_a_t (R^n): History of the last n (default is 3) steering angles.
        - h_d_t (R^n-1): History of delta (change in) steering values over the last n steps.
        """
        super().__init__(name, normalize)

        self.maxlen_history = maxlen_history
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3*4 + 2*self.maxlen_history - 1,),
            dtype=np.float32
        )


        self.last_velocity = np.array([0., 0., 0.], dtype=np.float32)
        self.last_time = 0

        self.angles: deque = deque([0.0] * self.maxlen_history, maxlen=self.maxlen_history)


    def _get_obs(self, game_states, **kwargs):
        # Extract simulation state data
        ssd: SimStateData = game_states[IPCFields.SIMSTATE]
        dyna_current: HmsDynaStateStruct = ssd.dyna.current_state

        # Orientation and velocities
        orientation = dyna_current.rotation.to_numpy().T.astype(np.float32)  # (3, 3)
        velocity = np.array(dyna_current.linear_speed, dtype=np.float32)     # (3,)
        angular_speed = np.array(dyna_current.angular_speed, dtype=np.float32)  # (3,)

        # Time delta
        time = ssd.time / constants.MILLISECONDS_TO_SECONDS
        delta_t = time - self.last_time

        # Compute linear velocity & acceleration, Angular velocit in car frame
        v_t: np.ndarray = orientation @ velocity
        a_t: np.ndarray = np.zeros(3, dtype=np.float32) if delta_t == 0 else (v_t - self.last_velocity) / delta_t
        v_r_t: np.ndarray = orientation @ angular_speed

        # Normalize control inputs to [-1, 1]
        throttle = ssd.scene_mobil.input_gas  #[0,1]
        brake = ssd.scene_mobil.input_brake  # [0,1]
        steer = ssd.scene_mobil.input_steer # [-1,1]
        c_t: np.ndarray = np.array([steer, throttle, brake], dtype=np.float32)

        # Steering angle and history
        turning_rate = ssd.scene_mobil.turning_rate  # [-1, 1]
        steering_angle = self.MAX_STEER_ANGLE * turning_rate

        self.angles.append(steering_angle)
        h_a_t = np.array(self.angles, dtype=np.float32)         # Steering angle history
        h_d_t = np.diff(h_a_t, axis=0) if len(h_a_t) > 1 else np.zeros(0, dtype=np.float32)  # Steering deltas

        # Update internal state
        self.last_time = time
        self.last_velocity = v_t

        # Debug info
        self.info.update({
            "linear_velocity": v_t,
            "linear_acceleration": a_t,
            "angular_velocity": v_r_t,
            "throttle": throttle,
            "brake": brake,
            "steer": steer,
            "history_steering_angles": h_a_t,
            "history_steering_deltas": h_d_t,
        })

        # Final proprioceptive observation
        propriocentric_features = np.hstack([
            v_t.ravel(),      # (3,)
            a_t.ravel(),      # (3,)
            v_r_t.ravel(),    # (3,)
            c_t.ravel(),      # (3,)
            h_a_t.ravel(),    # (n,)
            h_d_t.ravel(),    # (n-1,)
        ]).astype(np.float32)

        return propriocentric_features

    def _normalize(self, obs):
        """Normalize proprioceptive features."""

        i = 0
        v_t = obs[i:i+3] / self.MAX_LINEAR_SPEED;       i += 3
        a_t = obs[i:i+3] / self.MAX_ACCEL;              i += 3
        v_r_t = obs[i:i+3] / self.MAX_ANGULAR_SPEED;    i += 3

        c_t_raw = obs[i:i+3]
        steer = c_t_raw[0] # already in [-1, 1]
        throttle = 2.0 * c_t_raw[1] - 1.0             
        brake = 2.0 * c_t_raw[2] - 1.0
        c_t = np.array([steer, throttle, brake], dtype=np.float32)
        i += 3

        n = self.maxlen_history
        h_a_t = obs[i:i+n] / self.MAX_STEER_ANGLE;      i += n
        h_d_t = obs[i:] / self.MAX_STEER_DELTA          # Remaining elements

        normalized = np.concatenate([
            v_t, a_t, v_r_t, c_t, h_a_t, h_d_t
        ]).astype(np.float32)

        return normalized
    
    def reset(self):
        self.last_velocity = np.array([0.,0.,0.],dtype=np.float32)
        self.h_a_t = np.array([0.,0.,0.],dtype=np.float32)
        self.angles : deque = deque([0.]*self.maxlen_history, maxlen=self.maxlen_history)
    
class GlobalFeaturesTerm(ObservationTerm):
    def __init__(self, name ="global features", normalize = False,lookahead_sec:int = 6,n_points:int = 60):
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
        """
        #NOTE as for now we only return the center points no left and right
        super().__init__(name, normalize)
        self.lookahead_sec = lookahead_sec
        self.n_points = n_points
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_points * 3,),  # ← flat vector
            dtype=np.float32
        )

    def _get_obs(self, game_states, **kwargs):
        ssd: SimStateData = game_states[IPCFields.SIMSTATE]
        reference_line = self.env.reference_line
        # current dynamic state of the car, such as its position, orientation, speed ... .
        dyna_current: HmsDynaStateStruct = ssd.dyna.current_state
        
        position = np.array(dyna_current.position,dtype=np.float32,)  # (3,)
        orientation = dyna_current.rotation.to_numpy().T.astype(np.float32)  # (3, 3)
        speed = ssd.display_speed / constants.MS_TO_KMH

        next_idx, d, _ = reference_line.get_distance_to_next_point()
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

        self.info.update({
            "comming_refline_points" : comming_refline_points,
            "orientation": orientation,
            "position": position
        })
     
        return comming_refline_points.ravel().astype(np.float32) 
    
    def _normalize(self, obs):
        return obs