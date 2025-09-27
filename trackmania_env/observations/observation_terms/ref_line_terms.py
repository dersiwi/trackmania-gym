from typing import Union
import numpy as np
from gymnasium.spaces import Box
from tminterface.structs import SimStateData, HmsDynaStateStruct

from trackmania_env.observations.observation_manager import ObservationTerm
from trackmania_env.utils.constants import ObsNormalizationFactors
from game_interaction.ipc_fields import IPCFields
from trackmania_env.utils.reference_line_manager import ReferenceLineManager

class NextReflinePoint(ObservationTerm):
    """
    Returns a fixed number of upcoming reference line points,
    transformed into the car's local coordinate system.
    """

    def __init__(self,n_refline_points: int,reference_line_stride: int,normalize: bool = True,name: str = "static_ref_line_points"):
        super().__init__(name, normalize)
        self.n_refline_points = n_refline_points
        self.reference_line_stride = reference_line_stride

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_refline_points, 3),
            dtype=np.float32
        )

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

        return rel_points

class LateralDistance(ObservationTerm):

    def __init__(self,name="laterale distance",normalize = True):
        super().__init__(name, normalize)
        # TODO what are the bounds ?
        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(),
            dtype=np.float32
        )
    
    def _normalize(self, obs):
        return obs / ObsNormalizationFactors.lateral_dist_norm

    def _get_obs(self, game_states :  dict[str, Union[np.ndarray, SimStateData]], **kwargs):
        reference_line = self.env.reference_line
        next_idx, _, _ = reference_line.get_distance_to_next_point()
        lateral_distance : np.ndarray = reference_line.calculate_lateral_difference(next_idx, game_states[IPCFields.SIMSTATE].position)
        return lateral_distance
    
class RelativeDistance(ObservationTerm):
    def __init__(self,name="relative distance",normalize = True):
        super().__init__(name, normalize)
        # TODO what are the bounds ?
        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(),
            dtype=np.float32
        )
    def _normalize(self, obs):
        return obs
    
    def _get_obs(self, game_states:  dict[str, Union[np.ndarray, SimStateData]], **kwargs):
        reference_line = self.env.reference_line
        _, _, drel = reference_line.get_distance_to_next_point()
        return drel