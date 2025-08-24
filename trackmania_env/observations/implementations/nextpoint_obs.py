from __future__ import annotations
import traceback
from trackmania_env.observations.observation_manager import ObservationManager
from gymnasium import spaces
import numpy as np

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
from trackmania_env.observations.terms import NextReflinePoint, MobileStatesTerm, SurfaceFloats, SpeedTerm, ObservationTerm, LateralDistance
from trackmania_env.utils.contact_materials import physics_behavior_fromint,NUM_SURFACE_CATEGORIES


class NextPointObsManager(ObservationManager):
    def __init__(self, colorspace, convert_torch, img_width, img_height, normalize_obs,normalize_sanity_check = False):
        super().__init__(colorspace, convert_torch, img_width, img_height, normalize_obs=normalize_obs)
        self.reference_line_points_lookahead = 10
        self.refeence_line_stride = 10
        assert self.reference_line_points_lookahead % self.refeence_line_stride == 0


        self.nrp = NextReflinePoint(self.reference_line_points_lookahead, normalize_obs)
        self.speed = SpeedTerm(normalize_obs)
        self.mobile_states = MobileStatesTerm(normalize_obs)
        self.surfaces = SurfaceFloats(normalize_obs)
        self.ldist = LateralDistance(normalize_obs)
        
        self.all_obs_terms : list[ObservationTerm] = [self.nrp, self.speed, self.mobile_states, self.surfaces, self.ldist]

        self.statevector_dim = 1 # TODO Also create Term for drel.
        for term in self.all_obs_terms:
            self.statevector_dim += term.dim


    def get_observation_dict(self) -> spaces.Dict:
        """Returns observation dict for environment according to initialization."""
        return spaces.Dict({
                "image": spaces.Box(low=0, high=255, shape=(self.n_channels, self.img_height, self.img_width), dtype=np.uint8),
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(self.statevector_dim,), dtype=np.float32),
            })
    

    def get_values_from_state_dict(self, obs : SimStateData):

        next_idx, d, drel = self.env.reference_line.get_distance_to_next_point()

        comming_refline_points = self.env.reference_line.get_reference_line_points(begin_idx=next_idx,
                                                                end_idx= next_idx + self.reference_line_points_lookahead * self.refeence_line_stride, 
                                                                extrapolate= True, 
                                                                stride = self.refeence_line_stride)

        floatvec : np.ndarray = np.hstack([drel,
                                            self.speed.get_observation(obs),
                                            self.ldist.get_observation(obs, lateral_distance = self.env.reference_line.calculate_lateral_difference(next_idx, obs.position)),
                                            self.nrp.get_observation(obs, reflinepoints=comming_refline_points),
                                            self.mobile_states.get_observation(obs)], dtype =  np.float32)
        
        assert floatvec.shape[0] == self.statevector_dim, f"Floatvector has size {floatvec.shape[0]}, however should be size {self.statevector_dim}"
        
        self.info["position"] = obs.position
        self.info["d"] = d
        self.info["drel"] = drel
        self.info["comming_refline_points"] = comming_refline_points

        return floatvec

