from trackmania_env.rewards.reward_calculation import RewradCalculator
from tminterface.structs import CheckpointData, SimStateData, CheckpointTime
from game_interaction.ipc_fields import IPCFields
import numpy as np
import torch

MS_TO_KMH = 3.6       # meters per second to kilometers per hour
MILLISECONDS_TO_SECONDS = 1000  # milliseconds to seconds
MAX_DISTANCE_TO_REFLINE = 15 # the distance after which the car is considered to be off-course

class SophyRewards(RewradCalculator):

    def __init__(self, reward_cfg,maxlen_history:int = 3):
        self.last_time = 0
        self.last_lateral_contact_time = 0
        self.last_off_course_time = 0

        self.maxlen_history:int = maxlen_history # TODO add this to the config

        self.c_d = reward_cfg.c_d # threshold angle
        self.c_s = reward_cfg.c_s # sensitivity factor
        self.c_o = reward_cfg.c_o # offset value

        self.w_progress = reward_cfg.progress_weight
        self.w_off_course = reward_cfg.off_course_penalty_weight
        self.w_wall = reward_cfg.wall_penalty_weight
        self.w_steer_change = reward_cfg.steering_change_penalty_weight
        self.w_steer_history = reward_cfg.steering_history_penalty_weight

        super().__init__(reward_cfg)

    def reset(self):
        self.last_lateral_contact_time = 0
        #self.last_off_course_time = 0

    
    def calculate_reward(self, observations, processed_obs, race_finished, other_terminations):
        """
        https://arxiv.org/pdf/2406.12563v1

        Calculates the agent's reward at the current time step, following the reward design
        described in Wurman et al. (2022).

        The reward is formulated as a weighted sum of several components, each targeting
        a specific aspect of safe and efficient racing behavior:

        - Course progress (rp): Encourages forward progress by projecting the agent`s 
          position onto the center line of the track and measuring distance advanced.

        - Off-course penalty (ro): Penalizes the agent for having three or more tires outside 
          the track boundaries. The penalty is scaled by velocity and the duration spent off-track.

        - Wall penalty (rw): Penalizes collisions with track walls to discourage using walls 
          for steering or speed advantages. Also scaled by velocity and contact duration.

        - Steering change penalty (rs): Discourages abrupt steering changes by penalizing 
          large differences in steering angles between consecutive time steps.

        - Steering history penalty (rh): Penalizes inconsistent steering patterns within a 
          short time window to promote smoother driving behavior. This penalty uses a 
          thresholded and smoothed function over recent steering changes.

        The total reward at time t is computed as:
            r_t = w_p * rp_t + w_o * ro_t + w_w * rw_t + w_s * rs_t + w_h * rh_t


        Args:
            observations (dict): Raw input observations, including simulation state data.
            processed_obs: Preprocessed observations (e.g., features extracted from input data).
            race_finished (bool): Indicates whether the current episode has ended due to race completion.
            other_terminations (bool): Indicates if the episode has ended for any other reason (e.g., crash).

        Returns:
            float: The scalar reward value for the current time step.
        """
        ssD : SimStateData = observations[IPCFields.SIMSTATE]
        propriocentric_features = processed_obs["propriocentric_features"]
        time =  ssD.time/MILLISECONDS_TO_SECONDS
        delta_time = time - self.last_time

        # This value is in km/h. When manually computing speed by taking the norm of the 3D velocity vector,
        # the result matches the displayed speed after multiplying by 3.6 (to convert from m/s to km/h).
        speed = ssD.display_speed / MS_TO_KMH

        # progress
        next_refline_index, d, drel = self.refline_manager.get_distance_to_next_point()
        rp = d
        rp = rp* self.w_progress

        # off-course
        lateral_distance = self.refline_manager.calculate_lateral_difference(idx=next_refline_index,car_position=ssD.position)
        is_off_course = lateral_distance >= MAX_DISTANCE_TO_REFLINE
        off_course_time = self.last_off_course_time + (delta_time if is_off_course else 0)
        ro = -(off_course_time - self.last_off_course_time) * speed
        ro= ro* self.w_off_course

        # wall
        lateral_contact_time = ssD.scene_mobil.last_has_any_lateral_contact_time/ MILLISECONDS_TO_SECONDS
        rw = -(lateral_contact_time - self.last_lateral_contact_time) * speed
        rw = rw * self.w_wall

        # steering change
        # TODO: Consider creating a custom Sophy environment to avoid relying on manual indexing 
        # for field access. The current approach is unclear and can lead to confusion or bugs.
        steering_change = torch.abs(propriocentric_features[-1])
        rs = -steering_change * self.w_steer_change

        # Extract the last two steering deltas
        h_d_t_tail = propriocentric_features[-2:]
        has_sign_flip = torch.sign(propriocentric_features[1]) != torch.sign(propriocentric_features[0])
        above_threshold = (h_d_t_tail.abs() > self.c_d).all(dim=0)

        # Indicator for inconsistent steering behavior
        m_t = 1 if (has_sign_flip and above_threshold) else 0

        delta_t = torch.sum(torch.abs(h_d_t_tail))
        # Compute the steering history penalty
        rh = -m_t / (1 + torch.exp(-self.c_s * (delta_t - self.c_o)))
        rh = rh* self.w_steer_history

        rp = float(rp)
        ro = float(ro)
        rw = float(rw)
        rs = float(rs)
        rh = float(rh)

        reward = rp + ro + rw + rs + rh
        
        self.last_lateral_contact_time = lateral_contact_time
        self.last_time = time 
        self.last_off_course_time = off_course_time

        return reward, {"total": reward,
                        "progress": rp,
                        "off-course": ro,
                        "wall": rw,
                        "steering_changed": rs,
                        "steering_history": rh,
                        "nextpoint_reference_index" : next_refline_index,}


        