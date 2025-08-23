import numpy as np

from trackmania_env.rewards.reward_calculation import RewradCalculator
from trackmania_env.utils import constants
from trackmania_env.rewards.implementations.nextpointrewards import NextPointRewards
from tminterface.structs import  SimStateData
from game_interaction.ipc_fields import IPCFields

MIN_DRIFT_ANGULAR_VELOCITY = 30 
MIN_DRIFT_SPEED = 100 

class NextPointDriftReward(NextPointRewards):
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
                drift_reward_weight = 1,
                lateral_distance_mode = "gauss",
                mean = 0,
                sigma = 1,
                yshift = -1,
                multiplicator = 5,
                dist_scale = 0.3,
                normalize = False,
                **kwargs):
        super().__init__(accum_distance_weight, race_not_finished_weight, race_finished_reward_weight, other_termination_punishment, velocity_reward_weight, backward_weight, distance_to_center_weight, velocity_change_reward_weight, speed_reward_weight, lateral_distance_mode, mean, sigma, yshift, multiplicator, dist_scale, normalize, **kwargs)
        
        self.drift_reward_weight = drift_reward_weight

    def is_off_course(self,lateral_distance,next_point_idx,position):
        refline_z_coord = self.refline_manager.reference_line[next_point_idx][1]
        agent_z_coord = position[1]
        height_diff = abs(agent_z_coord-refline_z_coord)
        
        return lateral_distance >= constants.MAX_DISTANCE_TO_REFLINE or height_diff >= constants.MAX_HEIGHT_DIFERENCE 
    
    def get_sum_of_weighted_rewards(self, observations, processed_obs, race_finished, other_terminations):
        reward, info = super().get_sum_of_weighted_rewards(observations, processed_obs, race_finished, other_terminations)

        ssD : SimStateData = observations[IPCFields.SIMSTATE]
        next_refline_index , d, _ = self.refline_manager.get_distance_to_next_point()
        is_off_course =  self.is_off_course(d,next_refline_index,ssD.position)

        speed = ssD.display_speed
        v_local = ssD.rotation_matrix @ ssD.velocity
        lateral_velocity = np.abs(v_local[0])

        # calculate drift reward
        drift_reward = 0
        
        is_drifting = lateral_velocity > MIN_DRIFT_ANGULAR_VELOCITY
        is_fast_enough = speed > MIN_DRIFT_SPEED
        if (not is_off_course) and is_drifting and is_fast_enough:
            drift_reward = 1

        drift_reward *= self.drift_reward_weight
        info["drift_reward"] = drift_reward

        reward += drift_reward

        return reward,info