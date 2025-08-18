from __future__ import annotations
import traceback
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

from trackmania_env.observations.implementations.independent_obs import IndependentObservationManager
from trackmania_env.utils.contact_materials import get_normalized_surface_float
from trackmania_env.utils.constants import NormalizationFactors

from dataclasses import dataclass

@dataclass(frozen=True)
class OSV:
    """Utility class for indexes in statevector of Nextpoint obs manager. (actually i don't remember what OSV stands for, but there is an acronym: my best gues is Observation [?] Vector)
    This class is extremenly dependent on the implementation of NextPointObs, since the main
    purpose is to indicate the inicie of Nexpointobs-state-vector.
    """
    n_refline_points: int

    @property
    def drel_idx(self) -> int:
        return 0

    @property
    def speed_idx(self) -> int:
        return 1

    @property
    def lateral_dist_idx(self) -> int:
        return 2

    # Ranges as slices (half-open)
    @property
    def refline(self) -> slice:
        start = 3
        stop = 3 + self.n_refline_points * 3
        return slice(start, stop) 

    @property
    def sliding(self) -> slice:
        start = self.refline.stop
        stop = start + 4
        return slice(start, stop)

    @property
    def ground_contact(self) -> slice:
        start = self.sliding.stop
        stop = start + 4
        return slice(start, stop)

    @property
    def damper_absorb(self) -> slice:
        start = self.ground_contact.stop
        stop = start + 4
        return slice(start, stop)

    # Scalars that follow the ranges
    @property
    def gearbox_state_idx(self) -> int:
        return self.damper_absorb.stop

    @property
    def gear_idx(self) -> int:
        return self.gearbox_state_idx + 1

    @property
    def actual_rpm_idx(self) -> int:
        return self.gear_idx + 1

    @property
    def is_freewheeling_idx(self) -> int:
        return self.actual_rpm_idx + 1

    @property
    def surface_categories(self) -> slice:
        start = self.is_freewheeling_idx + 1
        stop  = start + 4
        return slice(start, stop)

    @property
    def zero_to_surfaces(self) -> slice:
        # everything from 0 up to end of surface categories
        return slice(0, self.surface_categories.start)

    def validate(self, length: int | None = None) -> None:
        prev = -1
        for s in [slice(self.drel_idx, self.drel_idx+1),
                  slice(self.speed_idx, self.speed_idx+1),
                  slice(self.lateral_dist_idx, self.lateral_dist_idx+1),
                  self.refline, self.sliding, self.ground_contact, self.damper_absorb,
                  slice(self.gearbox_state_idx, self.gearbox_state_idx+1),
                  slice(self.gear_idx, self.gear_idx+1),
                  slice(self.actual_rpm_idx, self.actual_rpm_idx+1),
                  slice(self.is_freewheeling_idx, self.is_freewheeling_idx+1),
                  self.surface_categories]:
            assert prev <= s.start <= s.stop, "Non-monotonic indexes"
            prev = s.stop
        if length is not None:
            assert self.surface_categories.stop <= length, "Statevector too short"
    
    @staticmethod
    def label_block(osv: OSV, idx: int) -> str:
        if idx == osv.drel_idx: return "drel"
        if idx == osv.speed_idx: return "speed"
        if idx == osv.lateral_dist_idx: return "lateral_dist"
        if idx in range(osv.refline.start, osv.refline.stop): return "refline"
        if idx in range(osv.sliding.start, osv.sliding.stop): return "sliding"
        if idx in range(osv.ground_contact.start, osv.ground_contact.stop): return "ground_contact"
        if idx in range(osv.damper_absorb.start, osv.damper_absorb.stop): return "damper_absorb"
        if idx == osv.gearbox_state_idx: return "gearbox_state"
        if idx == osv.gear_idx: return "gear"
        if idx == osv.actual_rpm_idx: return "actual_rpm"
        if idx == osv.is_freewheeling_idx: return "is_freewheeling"
        if idx in range(osv.surface_categories.start, osv.surface_categories.stop): return "surface_category"
        return "unknown"



class NextPointObsManager(IndependentObservationManager):
    def __init__(self, colorspace, convert_torch, img_width, img_height, normalize_obs,normalize_sanity_check = False):
        super().__init__(colorspace, convert_torch, img_width, img_height, normalize_obs=normalize_obs)
        self.reference_line_points_lookahead = 10
        self.refeence_line_stride = 10
        assert self.reference_line_points_lookahead % self.refeence_line_stride == 0
        self.last_obs : SimStateData = None

        self.statevector_dim = 0
        self.idxs = OSV(self.reference_line_points_lookahead)

        self.normalize_sanity_check = normalize_sanity_check


    def normalize_state_vector(self, obs: np.ndarray) -> np.ndarray:
        """Normalizes the state vector. This method implements regularization by deviding by the max value; due to booleans in the obsrvazations."""
        obs[self.idxs.speed_idx]            /= NormalizationFactors.speed_norm                  
        obs[self.idxs.refline]              /= NormalizationFactors.refline_norm                   
        obs[self.idxs.gearbox_state_idx]    /= NormalizationFactors.gearbox_norm                      
        obs[self.idxs.actual_rpm_idx]       /= NormalizationFactors.rpm_norm                 
        obs[self.idxs.lateral_dist_idx]     /= NormalizationFactors.lateral_dist_norm                       
        obs[self.idxs.gear_idx]             /= NormalizationFactors.gear_norm                        
        #obs[self.idxs.damper_absorb]       /= 0.15                     
        
        if self.normalize_sanity_check:
            try:
                # Sanity checks <- i like the name of this
                gonetol = 1.0 + 1e-4
                goneidxs = np.nonzero(obs >= gonetol)[0]
                gonelabels = [OSV.label_block(self.idxs, i) for i in goneidxs]
                assert np.all(obs <= gonetol), f"Expected observations to be normalized to [0,1] before surface categories. \
                    Got unnormalized values : {goneidxs, obs[goneidxs]}. These are in the {gonelabels}"

                ref_block = obs[self.idxs.refline]
                expected_ref_len = self.idxs.n_refline_points * 3
                assert ref_block.shape[0] == expected_ref_len, f"Expected {expected_ref_len} refline values, got {ref_block.shape[0]}."

                # surface_categories is half-open [start, stop), so stop should equal total length
                assert self.idxs.surface_categories.stop == obs.shape[0], f"Last index should be {obs.shape[0]-1}, but it's {self.idxs.surface_categories.stop-1}."
            except Exception as e:
                traceback.print_exc()
        return obs



    def get_observation_dict(self) -> spaces.Dict:
        """Returns observation dict for environment according to initialization."""
        self.statevector_dim = (
            # refline points + drel + speed + latera_dist
            self.mobile_states_dim + 3 * self.reference_line_points_lookahead + 3 
        )
        return spaces.Dict({
                "image": spaces.Box(low=0, high=255, shape=(self.n_channels, self.img_height, self.img_width), dtype=np.uint8),
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(self.statevector_dim,), dtype=np.float32),
            })
    

    def get_values_from_state_dict(self, obs : SimStateData):

        dyna_current: HmsDynaStateStruct = obs.dyna.current_state

        mobile_states = self.get_mobil_states(obs)
        next_idx, d, drel = self.env.reference_line.get_distance_to_next_point()
        lateral_dist = self.env.reference_line.calculate_lateral_difference(next_idx, obs.position)

        velocity_delta : np.ndarray =  np.array([0,0,0])
        if not self.last_obs == None:
            velocity_delta = np.array(obs.velocity) - np.array(self.last_obs.velocity)

        speed = obs.display_speed
        
        orientation = np.array(dyna_current.rotation.to_numpy(), dtype=float).T
        comming_refline_points = self.get_next_refline_points(next_idx, obs.position, orientation,obs=obs)


        self.last_obs = obs

        floatvec : np.ndarray = np.hstack([drel,
                                           # velocity_delta,
                                            speed,
                                            lateral_dist,
                                            comming_refline_points.ravel(),
                                            mobile_states], dtype =  np.float32)
        
        assert floatvec.shape[0] == self.statevector_dim, f"Floatvector has size {floatvec.shape[0]}, however should be size {self.statevector_dim}"
        
        self.info["position"] = obs.position
        self.info["orientation"] = orientation
        self.info["d"] = d
        self.info["drel"] = drel
        self.info["velocity_delta"] = velocity_delta
        self.info["comming_refline_points"] = comming_refline_points
        self.info["mobile_states"] = mobile_states
        self.info["lateral_dist"] = lateral_dist

        return floatvec


    def get_next_refline_points(self, next_refline_idx : int, car_position : np.ndarray, car_orientation : np.ndarray,obs:SimStateData=None) -> np.ndarray:
        comming_refline_points = self.env.reference_line.get_reference_line_points(begin_idx=next_refline_idx,
                                                                end_idx= next_refline_idx + self.reference_line_points_lookahead * self.refeence_line_stride, 
                                                                extrapolate= True, 
                                                                stride = self.refeence_line_stride)
        
        assert comming_refline_points.shape[0] == self.reference_line_points_lookahead, f"Expected to get {self.reference_line_points_lookahead} points from reflinemanager, got {comming_refline_points.shape[0]}"
        
        comming_refline_points_rel_to_car : np.ndarray = np.array(car_orientation).dot((comming_refline_points - np.array(car_position)).T).T # (n,3)
        #comming_refline_points_rel_to_car_flattened = comming_refline_points_rel_to_car.ravel()
        return comming_refline_points_rel_to_car
