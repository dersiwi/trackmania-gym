from collections import deque
from trackmania_env.rewards.reward_calculation import RewradCalculator
from tminterface.structs import CheckpointData, SimStateData, CheckpointTime
from game_interaction.ipc_fields import IPCFields
from trackmania_env.utils import constants
import numpy as np
import torch

class SophyRewards(RewradCalculator):

  def __init__(self,
                maxlen_history:int = 3,
                progress_weight:float= 0.014,
                off_course_penalty_weight:float = 0.034, 
                wall_penalty_weight:float = 10.0,
                steering_change_penalty_weight:float= 3.0,
                steering_history_penalty_weight:float= 5.0, 
                c_d :float = 0.014,
                c_s :float = 182.883569,
                c_o :float =  0.034,
                normalize : bool = False):
    """
    Initializes the GT Sophy-inspired reward function with tunable weights for each component.

    This reward function is based on a weighted sum of progress and penalties that shape 
    expert-like racing behavior. It follows the formulation in Wurman et al. (2022), where the 
    agent is rewarded for forward progress and penalized for undesirable driving behaviors 
    such as going off-course, hitting walls, and erratic steering.

    Args:
        maxlen_history (int, default=3)                     : Number of previous time steps used to compute history-based penalties (e.g., steering consistency).
        progress_weight (float, default=0.014)              : Weight for the course progress reward term , encouraging lap time minimization.
        off_course_penalty_weight (float, default=0.034)    : Weight for the off-course penalty term , discouraging shortcutting or corner cutting.
        wall_penalty_weight (float, default=10.0)           : Weight for the wall-hit penalty term , penalizing contact with track walls.
        steering_change_penalty_weight (float, default=3.0) : Weight for the steering change penalty term , discouraging abrupt steering.
        steering_history_penalty_weight (float, default=5.0): Weight for the steering history penalty term , penalizing inconsistent steering direction over short time spans.
        c_d (float, default=0.014)                          : Threshold steering angle beyond which history-based penalties apply.
        c_s (float, default=182.883569)                     : Sensitivity factor used in the sigmoid function for the steering history penalty.
        c_o (float, default=0.034)                          : Offset in the sigmoid function used in the steering history penalty.
      """
    super().__init__()
    self.last_time = 0
    self.last_lateral_contact_time = 0
    self.last_off_course_time = 0
    self.last_drel = 0
    self.maxlen_history:int = maxlen_history 

    self.c_d = c_d # threshold angle
    self.c_s = c_s # sensitivity factor
    self.c_o = c_o # offset value

    self.w_progress = progress_weight
    self.w_off_course = off_course_penalty_weight
    self.w_wall = wall_penalty_weight
    self.w_steer_change = steering_change_penalty_weight
    self.w_steer_history = steering_history_penalty_weight

    super().__init__(normalize)
    self.max_degree_radians = np.pi / 6 # equivalent to np.deg2rad(30) so the max steering angle is |30| degree 
    self.angles : deque = deque([0.]*self.maxlen_history, maxlen=self.maxlen_history) # History of the last three steering angles
    self.epsilon = 1e-6  # Tolerance for float comparisons

    self.track_lenght = None 

  def reset(self):
    self.last_lateral_contact_time = 0
    self.last_drel = 0
    self.angles : deque = deque([0.]*self.maxlen_history, maxlen=self.maxlen_history)

    #self.last_off_course_time = 0

    
  def get_sum_of_weighted_rewards(self, observations, processed_obs, race_finished, other_terminations):      
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
        time =  ssD.time/ constants.MILLISECONDS_TO_SECONDS
        delta_time = time - self.last_time

        self.track_lenght = self.track_lenght or np.sum(self.refline_manager.segment_lengths)
        # This value is in km/h. When manually computing speed by taking the norm of the 3D velocity vector,
        # the result matches the displayed speed after multiplying by 3.6 (to convert from m/s to km/h).
        speed = ssD.display_speed / constants.MS_TO_KMH

        # progress
        next_refline_index, d, drel = self.refline_manager.get_distance_to_next_point()
        rp = drel-self.last_drel
        rp = rp* self.w_progress * self.track_lenght 

        # off-course
        lateral_distance = d#self.refline_manager.calculate_lateral_difference(idx=next_refline_index,car_position=ssD.position)
        is_off_course = lateral_distance >= constants.MAX_DISTANCE_TO_REFLINE
        off_course_time = self.last_off_course_time + (delta_time if is_off_course else 0)
        ro = -(off_course_time - self.last_off_course_time) * speed
        ro= ro* self.w_off_course

        # wall
        lateral_contact_time = ssD.scene_mobil.last_has_any_lateral_contact_time/ constants.MILLISECONDS_TO_SECONDS
        rw = -(lateral_contact_time - self.last_lateral_contact_time) * speed
        rw = rw * self.w_wall

        # steering change
        # TODO: Consider creating a custom Sophy environment to avoid relying on manual indexing 
        # for field access. The current approach is unclear and can lead to confusion or bugs.

        turning_rate = ssD.scene_mobil.turning_rate # [-1,1]
        steering_angle = self.max_degree_radians * turning_rate # when wheels are fully to the right then turning rate is equal to 1 but they only rotate too roughly 30 degree.
        assert np.abs(steering_angle) <= self.max_degree_radians + self.epsilon, f"steering_angle out of range: {steering_angle}"
        self.angles.append(steering_angle)

        h_a_t:np.ndarray = torch.tensor(self.angles,dtype=torch.float)
        h_d_t:np.ndarray = h_a_t[1:] - h_a_t[:-1]

        steering_change = torch.abs(h_d_t[-1])
        rs = -steering_change * self.w_steer_change

        # Extract the last two steering deltas
        h_d_t_tail = h_d_t[-2:]
        has_sign_flip = torch.sign(h_d_t[-1]) != torch.sign(h_d_t[-2])
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
        self.last_drel = drel

        return super().normalize_reward(reward), {"total": reward,
                                                  "progress": rp,
                                                  "off-course": ro,
                                                  "wall": rw,
                                                  "steering_changed": rs,
                                                  "steering_history": rh,
                                                  "nextpoint_reference_index" : next_refline_index,}


        