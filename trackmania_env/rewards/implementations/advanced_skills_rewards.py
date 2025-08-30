from typing import List
import numpy as np

from trackmania_env.rewards.reward_calculation import RewradCalculator
from trackmania_env.utils import constants
from trackmania_env.rewards.implementations.nextpointrewards import NextPointRewards
from tminterface.structs import  SimStateData,SimulationWheel
from game_interaction.ipc_fields import IPCFields
from trackmania_env.rewards.reward_calculation import RewardTerm

class OffTrackConditionedRewardTerm(RewardTerm):
    """
    A helper class for reward terms that ensures no reward is given if the agent is off-track.
    This is to prevent exploits where the agent could perform actions off-track while still receiving rewards.
    """
    def __init__(self, weight, name):
        super().__init__(weight, name)
    
    def is_off_course(self,position):
        next_refline_index , d, _ = self.env.reference_line.get_distance_to_next_point()
        refline_z_coord = self.env.reference_line.reference_line[next_refline_index][1]
        agent_z_coord = position[1]
        height_diff = abs(agent_z_coord-refline_z_coord)
        return d >= constants.MAX_DISTANCE_TO_REFLINE or height_diff >= constants.MAX_HEIGHT_DIFERENCE

    def _on_track_reward(self, observations, processed_obs, race_finished, other_terminations):
        raise NotImplementedError
    
    def _get_term(self, observations, processed_obs, race_finished, other_terminations):
        position = observations[IPCFields.SIMSTATE].position
        if self.is_off_course(position): return 0
        return self._on_track_reward(observations, processed_obs, race_finished, other_terminations)
    
class DriftReward(OffTrackConditionedRewardTerm):
    def __init__(self, weight,min_drift_speed = 100, min_drift_angular_vel = 30, name="drift_reward"):
        super().__init__(weight, name)
        self.min_drift_angular_vel = min_drift_angular_vel
        self.min_drift_speed = min_drift_speed

    def _on_track_reward(self, observations, processed_obs, race_finished, other_terminations):
        ssD : SimStateData = observations[IPCFields.SIMSTATE]
    
        speed = ssD.display_speed
        v_local = ssD.rotation_matrix @ ssD.velocity
        lateral_velocity = np.abs(v_local[0])

        # calculate drift reward
        reward = 0
        is_drifting = lateral_velocity > self.min_drift_angular_vel
        is_fast_enough = speed > self.min_drift_speed
        if is_drifting and is_fast_enough:
            reward = 1

        return reward
    
class NextPointDriftReward(NextPointRewards):
    def __init__(self, 
                drift_reward_weight = 1,
                normalize = False,
                **kwargs):
        super().__init__(normalize=normalize, **kwargs)
        self.reward_terms.append(DriftReward(drift_reward_weight))

class AirBrakeNextPointReward(NextPointRewards):
    def __init__(self, 
                 accum_distance_weight = 1200,
                race_not_finished_weight = 0.05,
                race_finished_reward_weight = 100, 
                other_termination_punishment = 20,
                velocity_reward_weight = 1,
                backward_weight = 0,
                distance_to_center_weight = 1,
                velocity_change_reward_weight = 30,
                speed_reward_weight = 1,
                air_brake_reward_weight = 1,
                lateral_distance_mode = "gauss",
                mean = 0,
                sigma = 1,
                yshift = -1,
                multiplicator = 5,
                dist_scale = 0.3,
                normalize = False,
                **kwargs):
        super().__init__(accum_distance_weight, race_not_finished_weight, race_finished_reward_weight, other_termination_punishment, velocity_reward_weight, backward_weight, distance_to_center_weight, velocity_change_reward_weight, speed_reward_weight, lateral_distance_mode, mean, sigma, yshift, multiplicator, dist_scale, normalize, **kwargs)
        
        self.air_brake_reward_weight = air_brake_reward_weight
        self.mid_air_brake = False

    def reset(self):
        super().reset()
        self.mid_air_brake = False
    
    def get_sum_of_weighted_rewards(self, observations, processed_obs, race_finished, other_terminations):
        reward,info = super().get_sum_of_weighted_rewards(observations, processed_obs, race_finished, other_terminations)

        ssD : SimStateData = observations[IPCFields.SIMSTATE]
        wheels: np.ndarray[SimulationWheel] = ssD.simulation_wheels
        in_air:bool = all([not (wheels[i].real_time_state.has_ground_contact) for i in range(wheels.shape[0])])
        braking:bool = ssD.input_brake

        # air-brake reward
        air_brake_rew = 0
        if in_air:
            # only give reward for first brake in the air so that he doesn not spam it and gets slower
            if braking and not self.mid_air_brake:
                air_brake_rew = 1
                self.mid_air_brake = True
        else:
            # Reset when agent lands
            self.mid_air_brake = False

        air_brake_rew *= self.air_brake_reward_weight
        info["air_brake_reward"] = air_brake_rew
        reward += air_brake_rew

        return reward,info


from trackmania_env.utils.contact_materials import physics_behavior_fromint,SurfaceCategory

class SpeedSplideNextPointReward(NextPointRewards):
    
    # [forward_speed_kmh, max_side_friction]
    MAX_SIDE_FRICTION_FROM_SPEED = np.array([
        [0,   80],  
        [100, 80],   
        [200, 75],   
        [300, 67],  
        [400, 60], 
        [500, 55]
    ])

    # material_max_side_friction_multiplier | credits : https://github.com/TomashuTTTT7/TM-AlgoCrack/blob/main/cracks/sd_quality.as
    # for platform and road: 1.0
    # for dirt:              0.25
    # for grass:             0.15
    # for air:               0.0
    MATERIAL_SIDE_FRICTION_MULTIPLIER = {
        SurfaceCategory.ASPHALT: 1.0,
        SurfaceCategory.DIRT: 0.25,
        SurfaceCategory.GRASS: 0.15,
        # as for now we dont know what the other multipliers should be
        SurfaceCategory.TURBO: 0, # maybe just set it to 1 since it is somewhat like asphalt?
        SurfaceCategory.OTHER: 0,

    }

    def __init__(self, 
                 accum_distance_weight = 1200,
                race_not_finished_weight = 0.05,
                race_finished_reward_weight = 100, 
                other_termination_punishment = 20,
                velocity_reward_weight = 1,
                backward_weight = 0,
                distance_to_center_weight = 1,
                velocity_change_reward_weight = 30,
                speed_reward_weight = 1,
                speed_slide_reward_weight = 1,
                lateral_distance_mode = "gauss",
                mean = 0,
                sigma = 1,
                yshift = -1,
                multiplicator = 5,
                dist_scale = 0.3,
                normalize = False,
                **kwargs):
        super().__init__(accum_distance_weight, race_not_finished_weight, race_finished_reward_weight, other_termination_punishment, velocity_reward_weight, backward_weight, distance_to_center_weight, velocity_change_reward_weight, speed_reward_weight, lateral_distance_mode, mean, sigma, yshift, multiplicator, dist_scale, normalize, **kwargs)
        
        self.speed_slide_reward_weight = speed_slide_reward_weight
    
    def get_sum_of_weighted_rewards(self, observations, processed_obs, race_finished, other_terminations):

        reward,info = super().get_sum_of_weighted_rewards(observations, processed_obs, race_finished, other_terminations)

        ssD : SimStateData = observations[IPCFields.SIMSTATE]
        wheels: np.ndarray[SimulationWheel] = ssD.simulation_wheels
        in_air:bool = all([not (wheels[i].real_time_state.has_ground_contact) for i in range(wheels.shape[0])])

        speed_slide_reward = 0

        if not in_air:
            rotation = ssD.rotation_matrix
            linear_speed = ssD.dyna.current_state.linear_speed.to_numpy()
            speed = rotation.T @ linear_speed
            # speed[0] (x): speed sidewards
            # speed[1] (y): speed upwards
            # speed[2] (z): speed forwards
    
            wheels: np.ndarray[SimulationWheel] = ssD.simulation_wheels
            wheels_contact_material: List[int] = [wheel.real_time_state.contact_material_id for wheel in wheels]
            #the matrials ids are somehow messed up for the other tires
            front_tire_mat_id, rear_tire_mat_id = wheels_contact_material[1], wheels_contact_material[2]
            multip  = self.get_combined_side_friction_multiplier(front_tire_mat_id,rear_tire_mat_id)
   
            speed_slide_quality = self.get_speed_slide_quality(speed[0],speed[2],multip)
            print(f"{speed_slide_quality=}")
            speed_slide_reward = speed_slide_quality
        
        speed_slide_reward *= self.speed_slide_reward_weight

        info["speed_slide_reward"] = speed_slide_reward
        reward += speed_slide_reward

        return reward,info
    
    # credits to https://github.com/TomashuTTTT7/TM-AlgoCrack/blob/main/cracks/speedslide_quality.py
    def get_speed_slide_quality(self,x_speed:float,z_speed:float,material_max_side_friction_multiplier = 1.0):
        # look up max friction based on forward speed
        maxsidefriction = np.interp(constants.MS_TO_KMH * z_speed ,self.MAX_SIDE_FRICTION_FROM_SPEED[:,0], self.MAX_SIDE_FRICTION_FROM_SPEED[:,1]) 
        maxsidefriction *= material_max_side_friction_multiplier
        sidefriction = 20 * abs(x_speed)
        if sidefriction > maxsidefriction and maxsidefriction != 0:
            return (sidefriction - maxsidefriction) / maxsidefriction
        return 0.0
    
    def get_combined_side_friction_multiplier(self, front_mat_id, rear_mat_id):
        front_category = physics_behavior_fromint[front_mat_id]
        rear_category = physics_behavior_fromint[rear_mat_id]

        front_mul = self.MATERIAL_SIDE_FRICTION_MULTIPLIER[front_category]
        rear_mul = self.MATERIAL_SIDE_FRICTION_MULTIPLIER[rear_category]

        return 0.5 * front_mul + 0.5 * rear_mul