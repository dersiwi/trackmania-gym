from __future__ import annotations
from trackmania_env.utils.position_buffer import PositionBuffer
from tminterface.structs import CheckpointData, SimStateData, CheckpointTime
from game_interaction.ipc_fields import IPCFields
import numpy as np
from numba import jit

from trackmania_env.rewards.reward_calculation import RewradCalculator
from trackmania_env.utils.speedslide_quality import speedslide_quality_tarmac


class LinesightRewardCalculator(RewradCalculator):
    def __init__(self,
                 constant_reward_per_ms: float =  -6 / 5000,
                 ms_per_action: int = 50,
                 reward_per_m_advanced_along_centerline: float = 5/500,
                 final_speed_reward_as_if_duration_s: float = 0,
                 engineered_speedslide_reward: float = 1,
                 engineered_neoslide_reward: float = 1,
                 engineered_kamikaze_reward: float = 1,
                 engineered_close_to_vcp_reward: float = 1,
                 engineered_reward_min_dist_to_cur_vcp: float = 5,
                 engineered_reward_max_dist_to_cur_vcp: float = 25):
        """
        Initializes the LinesightRewardCalculator with a set of weights and constants that define how
        rewards are computed during training or simulation.

        Args:
            constant_reward_per_ms (float): Time-based penalty or reward applied each millisecond of simulation.
                                            Negative values discourage long runtimes.
            ms_per_action (int): Time interval (in milliseconds) per action step.
            reward_per_m_advanced_along_centerline (float): Reward for each meter advanced along the centerline.
            final_speed_reward_as_if_duration_s (float): Artificial time factor used to weight the final speed reward.
                                                         Used to estimate final reward from current speed.
            engineered_speedslide_reward (float): Multiplier for rewarding specific sliding behavior (e.g. controlled drifts).
            engineered_neoslide_reward (float): Multiplier for encouraging a certain style of slide (Neo-style).
            engineered_kamikaze_reward (float): Multiplier for discouraging aggressive, reckless behavior.
            engineered_close_to_vcp_reward (float): Reward for staying close to visible checkpoints (VCPs).
            engineered_reward_min_dist_to_cur_vcp (float): Minimum distance for the VCP-based reward to activate.
            engineered_reward_max_dist_to_cur_vcp (float): Maximum distance for VCP reward to apply.
        """
        super().__init__()

        self.constant_reward_per_ms = constant_reward_per_ms
        self.ms_per_action = ms_per_action

        self.reward_per_m_advanced_along_centerline = reward_per_m_advanced_along_centerline
        self.final_speed_reward_as_if_duration_s = final_speed_reward_as_if_duration_s
        self.final_speed_reward_per_m_per_s = (
            self.reward_per_m_advanced_along_centerline * self.final_speed_reward_as_if_duration_s
        )

        self.engineered_speedslide_reward = engineered_speedslide_reward
        self.engineered_neoslide_reward = engineered_neoslide_reward
        self.engineered_kamikaze_reward = engineered_kamikaze_reward
        self.engineered_close_to_vcp_reward = engineered_close_to_vcp_reward
        self.engineered_reward_min_dist_to_cur_vcp = engineered_reward_min_dist_to_cur_vcp
        self.engineered_reward_max_dist_to_cur_vcp = engineered_reward_max_dist_to_cur_vcp

        self.last_obs : SimStateData = None
        self.last_meters_advanced_along_centerline = 0
        self.last_zone_centers = None


    def calculate_reward(self, observations, processed_obs : dict[str, any], race_finished, other_terminations):
        ssD : SimStateData = observations[IPCFields.SIMSTATE]
        ssD.simulation_wheels[0]
        wheel_state = [ssD.simulation_wheels[i].real_time_state for i in range(4)]
       
        """* (
            ms_per_action
            if (i < n_frames - 1 or ("race_time" not in rollout_results))
            else rollout_results["race_time"] - (n_frames - 2) * config_copy.ms_per_action
        )"""
        
        meters_advanced_along_centerline_rew = 0
        v_x, v_z = ssD.velocity[1], ssD.velocity[2] # v_x is actually index 1
        velocity_change_reward = 0
        if not self.last_obs == None:

            meters_advanced_along_centerline_rew = (self.env.info["meters_advanced_along_centerline"] - self.last_meters_advanced_along_centerline) * self.reward_per_m_advanced_along_centerline
            if self.final_speed_reward_per_m_per_s != 0 and v_z > 0:
                # car has velocity *forward*
                velocity_change_reward = self.final_speed_reward_per_m_per_s * (np.linalg.norm(ssD.velocity) - np.linalg.norm(self.last_obs.velocity))



        speedslide_reward = 0
        if self.engineered_speedslide_reward != 0 and all((ws.has_ground_contact for ws in wheel_state)):
            # all wheels touch the ground
            speedslide_reward = self.engineered_speedslide_reward * max(0.0, 1 - abs(speedslide_quality_tarmac(v_x, v_z) - 1))

        # lateral speed is higher than 2 meters per second
        lateral_speed_reward = self.engineered_neoslide_reward if abs(v_x) >= 2.0 else 0


        # kamikaze reward
        kamikaze_reward = 0
        if ( any((ws.has_ground_contact for ws in wheel_state))): #TODO : # engineered_kamikaze_reward != 0 and rollout_results["actions"][i] <= 2 or
            kamikaze_reward = self.engineered_kamikaze_reward

        too_close_to_vcp = 0
        if self.engineered_close_to_vcp_reward != 0:
            too_close_to_vcp = self.engineered_close_to_vcp_reward * max(
                self.engineered_reward_min_dist_to_cur_vcp,
                min(self.engineered_reward_max_dist_to_cur_vcp, np.linalg.norm(self.env.info["state_zone_center_coordinates_in_car_reference_system"])),
            )
        reward = self.constant_reward_per_ms + meters_advanced_along_centerline_rew 
        reward += velocity_change_reward + kamikaze_reward + lateral_speed_reward + too_close_to_vcp
        reward += speedslide_reward
        
        self.last_obs = observations
        self.last_meters_advanced_along_centerline = self.env.info["meters_advanced_along_centerline"]
        return reward, {"const_rew_per_ms" : self.constant_reward_per_ms,
                        "meters_advanced_along_centerline_rew" : meters_advanced_along_centerline_rew,
                        "velocity_change_reward" : velocity_change_reward,
                        "kamikaze_reward" : kamikaze_reward,
                        "lateral_speed_reward" : lateral_speed_reward,
                        "too_close_to_vcp" : too_close_to_vcp,
                        "speedslide_reward" : speedslide_reward}


