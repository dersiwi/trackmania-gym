from collections import deque

import numpy as np
import gymnasium as gym

from trackmania_env.observations.observation_term import ObservationTerm
from trackmania_env.utils import constants
from game_interaction.ipc_fields import IPCFields
from tminterface.structs import SimStateData, HmsDynaStateStruct

from trackmania_env.utils.map_loader import (
    load_map_with_extrapolated_centers,
    precalculate_virtual_checkpoints_information,
    sync_virtual_and_real_checkpoints)

class LinesightDynamicTerm(ObservationTerm):
    """
    Represents the current dynamic state of the car, such as its position, 
    orientation, speed, and angular velocity. All relevant vectors are transformed 
    into the car's local reference frame to help the agent learn from a consistent 
    and generalized input space.
    """
    Y_VECTOR = np.array([0, 1, 0], dtype=np.float32)

    def __init__(self, name ="linesight dynamic", normalize = False):
        super().__init__(name, normalize)
        self.observation_space = gym.spaces.Box(
            low= -np.inf,
            high= np.inf,
            shape= (9,),
            dtype= np.float32
        )

    def _get_obs(self, game_states, **kwargs):
        ssd: SimStateData = game_states[IPCFields.SIMSTATE]
        dyna_current: HmsDynaStateStruct = ssd.dyna.current_state

        orientation = dyna_current.rotation.to_numpy().T                     # (3, 3)
        velocity = np.array(dyna_current.linear_speed, dtype=np.float32)     # (3,)
        angular_speed = np.array(dyna_current.angular_speed, dtype=np.float32)  # (3,)

        # Transform world-frame vectors into the car's local reference frame
        # From TMInterface Discord:
        #   - X: left
        #   - Y: up
        #   - Z: forward

        y_map_vector_local = orientation.dot(self.Y_VECTOR)       # (3,)
        velocity_local = orientation.dot(velocity)                # (3,)
        angular_velocity_local = orientation.dot(angular_speed)   # (3,)

        # Concatenate observations into a single vector
        obs = np.concatenate([
            y_map_vector_local,
            velocity_local,
            angular_velocity_local
        ])  # Shape: (9,)

        return obs

    def _normalize(self, obs):
        return obs # as for now we dont know how we should normalise


class ZoneCenterFeatures(ObservationTerm):
    """this observationt erm handels calculating the upcoming zone center aka referecen lien point so to say """
    def __init__(self, 
                 name="linesight zone_centers", 
                 normalize = False, 
                 n_extrapolate_before:int = 20, 
                 n_extrapolate_after:int = 1000,
                 refline_path: str = None,
                 map_path: str = None,
                 sync_virt_and_real_cp:bool = True,
                 dist_between_vcp:float = 0.5,
                 road_width :int = 90,
                 zone_input_count:int = 40,
                 zone_stride:int = 20,):
        """
        - n_extrapolate_before (int, default=20): Number of synthetic zone centers to extrapolate before the start of the map.
        - n_extrapolate_after (int, default=1000) : Number of synthetic zone centers to extrapolate beyond the end of the actual map.
        - refline_path (str): path of the reference-line / center-line of the track file. Typically a .npy file
        - map_path(str): path of the map file (GBX file)
        - sync_virt_and_real_cp(bool): flag to sync virtual checkpoints (reference line) with real ingame checkpoints 
        - dist_between_vcp (float, default=0.5) : Distance in meters between consecutive virtual checkpoints.
        - road_width(int): width of the road
        - zone_input_count (int, default=40): Number of zone centers to include in the observation input.
        - zone_stride (int, default=20) : Sampling interval; use one zone center every N centers in the input.
        """
        super().__init__(name, normalize)

        self.observation_space = gym.spaces.Box(
            low= -np.inf,
            high= np.inf,
            shape= (3*zone_input_count,),
            dtype= np.float32,
        )
        # Load the zone centers with extrapolation
        self.zone_centers: np.ndarray = load_map_with_extrapolated_centers(
            n_before      = n_extrapolate_before,
            n_after       = n_extrapolate_after,
            path_to_file  = refline_path,
        )

        # Precalculate information for virtual checkpoints
        (
            self.zone_transitions,
            self.dist_between_zones,
            self.dist_from_start_to_zone,
            self.normalized_vec_along_track,
        ) = precalculate_virtual_checkpoints_information(self.zone_centers)

        (
            self.next_real_cp_pos,
            self.max_dist_to_real_cp,
        ) = sync_virtual_and_real_checkpoints(
            zone_centers = self.zone_centers,
            map_path     = map_path,
            sync_virtual_and_real_checkpoints = sync_virt_and_real_cp,
        )

        self.n_extrapolate_before = n_extrapolate_before
        self.n_extrapolate_after = n_extrapolate_after
        self.current_zone_idx = self.n_extrapolate_before
        self.max_dist_to_virtual_cp = np.sqrt((dist_between_vcp/ 2) ** 2 + (road_width / 2) ** 2)
        self.dist_since_start:int = 0
        self.zone_stride = zone_stride
        self.zone_input_count = zone_input_count

    def reset(self):
        self.dist_since_start:int = 0
        self.current_zone_idx = self.n_extrapolate_before
    
    def _normalize(self, obs):
        return obs
    
    def _get_obs(self, game_states, **kwargs):
        ssd: SimStateData = game_states[IPCFields.SIMSTATE]
        dyna_current: HmsDynaStateStruct = ssd.dyna.current_state

        position = np.array(dyna_current.position,dtype=np.float32)  # (3,)
        orientation = dyna_current.rotation.to_numpy().T  # (3, 3)

        # Compute progress within the current zone 
        prev_idx = self.current_zone_idx - 1
        zone_vec = self.normalized_vec_along_track[prev_idx]
        zone_start = self.zone_transitions[prev_idx]
        zone_len = self.dist_between_zones[prev_idx]
        dist_in_zone = np.clip((position - zone_start).dot(zone_vec), 0, zone_len)

        # Total distance from start of track
        self.dist_since_start = self.dist_from_start_to_zone[prev_idx] + dist_in_zone

        if position[1] > -np.inf:
            self.current_zone_idx = self.update_zone_idx(
                idx= self.current_zone_idx,
                zones= self.zone_centers,
                pos= position,
                max_dist_virtual= self.max_dist_to_virtual_cp,
                real_cp= self.next_real_cp_pos,
                max_dist_real= self.max_dist_to_real_cp,
            )

        # Convert upcomin zone centers positions to local (car) frame 
        # Select a slice of zone centers starting at the current index, spaced by `step`
        start = self.current_zone_idx
        step = self.zone_stride
        count = self.zone_input_count
        end = start + step * count

        future_zone_slice = self.zone_centers[start:end:step, :]  # (count, 3)
        relative_zone_positions = (future_zone_slice - position).T
        zone_centers_local = orientation.dot(relative_zone_positions).T  # (count, 3)
        
        return np.array(zone_centers_local, dtype=np.float32).ravel(), {}
    
    
    def update_zone_idx(self,
        idx: int,
        zones: np.ndarray,
        pos: np.ndarray,
        max_dist_virtual: float,
        real_cp: np.ndarray,
        max_dist_real: np.ndarray) -> int:
        d_next = np.linalg.norm(zones[idx + 1] - pos)
        d_curr = np.linalg.norm(zones[idx] - pos)
        d_prev = np.linalg.norm(zones[idx - 1] - pos)
        d_real = np.linalg.norm(real_cp[idx] - pos)

        # Forward movement through virtual zones
        while (
            d_next <= d_curr and
            d_next <= max_dist_virtual and
            idx < len(zones) - 1 - self.n_extrapolate_after and
            d_real < max_dist_real[idx]
        ):
            idx += 1
            d_curr, d_prev = d_next, d_curr
            d_next = np.linalg.norm(zones[idx + 1] - pos)
            d_real = np.linalg.norm(real_cp[idx] - pos)

        # Backward movement if closer to previous zone
        while (
            idx >= 2 and
            d_prev < d_curr and
            d_prev <= max_dist_virtual
        ):
            idx -= 1
            d_next, d_curr = d_curr, d_prev
            d_prev = np.linalg.norm(zones[idx - 1] - pos)

        return idx
        
