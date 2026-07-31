from typing import Union
import numpy as np
from gymnasium.spaces import Box
from tminterface.structs import SimStateData, HmsDynaStateStruct

from trackmania_gym.trackmania_env.envs.info import EnvironmentInfo
from trackmania_gym.trackmania_env.observations.observation_term import ObservationTerm, VectorlikeTerm
from trackmania_gym.trackmania_env.utils.constants import ObsNormalizationFactors
from trackmania_gym.game_interaction.ipc_fields import IPCFields
from trackmania_gym.trackmania_env.utils.reference_line_manager import ReferenceLineManager

class NextReflinePoint(ObservationTerm):
    """
    Returns a fixed number of upcoming reference line points,
    transformed into the car's local coordinate system.
    """

    def __init__(self,n_refline_points: int,reference_line_stride: int,normalize: bool = True,name: str = "static_ref_line_points"):
        super().__init__(name, normalize)
        assert n_refline_points % reference_line_stride == 0
        self.n_refline_points = n_refline_points
        self.reference_line_stride = reference_line_stride

        self.set_observation_space_as_box(-1000, 1000, (n_refline_points, 3))

        
    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        return obs / ObsNormalizationFactors.refline_norm

    def _get_obs(self,game_states: dict[str, Union[np.ndarray, SimStateData]],**kwargs) -> np.ndarray:
        ssd: SimStateData = game_states[IPCFields.SIMSTATE]
        dyna_current: HmsDynaStateStruct = ssd.dyna.current_state

        car_position = ssd.position
        car_orientation = np.array(dyna_current.rotation.to_numpy(), dtype=float).T

        refline: ReferenceLineManager = self.env.reference_line
        next_idx, _, _ = refline.get_distance_to_next_point()

        upcoming_refline_points: np.ndarray = refline.get_reference_line_points(
            begin_idx=next_idx,
            end_idx=next_idx + self.n_refline_points * self.reference_line_stride,
            extrapolate=True,
            stride=self.reference_line_stride
        )

        assert upcoming_refline_points.shape[0] == self.n_refline_points, (
            f"Expected {self.n_refline_points} refline points, "
            f"but got {upcoming_refline_points.shape[0]}"
        )

        # Transform points into car's local frame
        rel_points: np.ndarray = car_orientation.dot(
            (upcoming_refline_points - car_position).T
        ).T  # Shape: (n_refline_points, 3)

        self.info[EnvironmentInfo.COMING_REFLINE_POINTS] = rel_points
        self.info[EnvironmentInfo.ORIENTATION] = car_orientation
        return rel_points, self.info
    
    def flatten(self, processed_obs):
        return processed_obs.reshape((processed_obs.shape[0] * processed_obs.shape[1]))
    
    def get_flatten_dim(self):
        return self.n_refline_points * 3
    
    def get_native_shape(self):
        return (self.n_refline_points, 3)

class LateralDistance(VectorlikeTerm):
    """Lateral Distance to the centerline. This is measured against the current last reference line point."""

    def __init__(self, name="laterale distance", normalize = True):
        super().__init__(name, normalize, 1) # TODO Bounds not great

    def _normalize(self, obs):
        return obs / ObsNormalizationFactors.lateraltrack_dist_norm

    def _get_obs(self, game_states :  dict[str, Union[np.ndarray, SimStateData]], **kwargs):
        reference_line = self.env.reference_line
        next_idx, _, _ = reference_line.get_distance_to_next_point()
        lateral_distance : np.ndarray = reference_line.calculate_lateral_difference(next_idx, game_states[IPCFields.SIMSTATE].position)
        return np.array(lateral_distance,dtype=np.float32), {}
    
class RelativeDistance(VectorlikeTerm):
    def __init__(self,name="relative distance",normalize = True):
        super().__init__(name, normalize, 1) # TODO what are the bounds ?
        
    def _normalize(self, obs):
        return obs
    
    def _get_obs(self, game_states:  dict[str, Union[np.ndarray, SimStateData]], **kwargs):
        reference_line = self.env.reference_line
        _, _, drel = reference_line.get_distance_to_next_point()
        return np.array(drel,np.float32), {}
