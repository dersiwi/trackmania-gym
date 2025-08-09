

from trackmania_env.observations.observation_manager import ObservationManager
from gymnasium import spaces
import numpy as np
from configs.config import LinesightObsCfg

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
from trackmania_env.utils.contact_materials import physics_behavior_fromint,NUM_SURFACE_CATEGORIES

from dataclasses import dataclass

@dataclass(frozen=True)
class OSV:
    """Utility class for indexes in statevector of Nextpoint obs manager."""
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
        stop  = start + 4 * NUM_SURFACE_CATEGORIES
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
"""
    def label_block(idx):
        if idx in range(self.idxs.refline.start, self.idxs.refline.stop): return "refline"
        if idx in range(self.idxs.sliding.start, self.idxs.sliding.stop): return "sliding"
        if idx in range(self.idxs.ground_contact.start, self.idxs.ground_contact.stop): return "ground_contact"
        if idx in range(self.idxs.damper_absorb.start, self.idxs.damper_absorb.stop): return "damper_absorb"
        if idx == self.idxs.speed_idx: return "speed"
        if idx == self.idxs.gear_idx: return "gear"
        if idx == self.idxs.gearbox_state_idx: return "gearbox_state"
        if idx == self.idxs.actual_rpm_idx: return "rpm"
        if idx == self.idxs.is_freewheeling_idx: return "is_freewheeling"
        return "unknown"
labels = [label_block(i) for i in bad_abs]
print(list(zip(bad_abs, labels, bad_vals)))
"""

class NextPointObsManager(ObservationManager):
    def __init__(self, colorspace, convert_torch, img_width, img_height, normalize_obs):
        super().__init__(colorspace, convert_torch, img_width, img_height, normalize_obs=normalize_obs)
        self.reference_line_points_lookahead = 10
        self.refeence_line_stride = 10
        assert self.reference_line_points_lookahead % self.refeence_line_stride == 0
        self.last_obs : SimStateData = None

        self.statevector_dim = 0
        self.idxs = OSV(self.reference_line_points_lookahead)

    def label_block(self, idx):
        if idx in range(self.idxs.refline.start, self.idxs.refline.stop): return "refline"
        if idx in range(self.idxs.sliding.start, self.idxs.sliding.stop): return "sliding"
        if idx in range(self.idxs.ground_contact.start, self.idxs.ground_contact.stop): return "ground_contact"
        if idx in range(self.idxs.damper_absorb.start, self.idxs.damper_absorb.stop): return "damper_absorb"
        if idx == self.idxs.speed_idx: return "speed"
        if idx == self.idxs.gear_idx: return "gear"
        if idx == self.idxs.gearbox_state_idx: return "gearbox_state"
        if idx == self.idxs.actual_rpm_idx: return "rpm"
        if idx == self.idxs.is_freewheeling_idx: return "is_freewheeling"
        if idx == self.idxs.lateral_dist_idx: return "lateral distance"
        return "unknown"

    def normalize_state_vector(self, obs: np.ndarray) -> np.ndarray:

        obs[self.idxs.speed_idx] /= 1000.0                        # speed
        obs[self.idxs.refline]   /= 500.0                         # refline points (this should actually normalize by 1000)
        obs[self.idxs.gearbox_state_idx] /= 5.0                   # gearbox
        obs[self.idxs.actual_rpm_idx]   /= 12000.0                # rpm (dont know what the correct max-value is)
        obs[self.idxs.damper_absorb]    /= 0.15                   # damper absorb
        obs[self.idxs.lateral_dist_idx] /= 15                     # lateral distance.

        # Sanity checks <- i like the name of this
        gonetol = 1.0 + 1e-4
        goneidxs = np.nonzero(obs[self.idxs.zero_to_surfaces] >= gonetol)
        gonelabels = [self.label_block(i) for i in goneidxs[0]]
        assert np.all(obs[self.idxs.zero_to_surfaces] <= gonetol), f"Expected observations to be normalized to [0,1] before surface categories. \
              Got unnormalized values : {goneidxs, obs[goneidxs]}. These are in the {gonelabels}"

        ref_block = obs[self.idxs.refline]
        expected_ref_len = self.idxs.n_refline_points * 3
        assert ref_block.shape[0] == expected_ref_len, f"Expected {expected_ref_len} refline values, got {ref_block.shape[0]}."

        # surface_categories is half-open [start, stop), so stop should equal total length
        assert self.idxs.surface_categories.stop == obs.shape[0], f"Last index should be {obs.shape[0]-1}, but it's {self.idxs.surface_categories.stop-1}."
        return obs



    def get_observation_dict(self) -> spaces.Dict:
        """Returns observation dict for environment according to initialization."""
        self.statevector_dim = (
            #mobile states 
            4*NUM_SURFACE_CATEGORIES
            + 3*4
            + 4*1
            # refline points
            + 3 * self.reference_line_points_lookahead
            # others
            + 3*1 # drel + speed + latera_dist
        )
       # 5 + 3 * self.reference_line_points_lookahead \
       #     + 4*NUM_SURFACE_CATEGORIES + 3*3 + 4*1 +1 # get_mobile_states 


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
        comming_refline_points = self.get_next_refline_points(next_idx, obs.position, orientation)


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


    def get_next_refline_points(self, next_refline_idx : int, car_position : np.ndarray, car_orientation : np.ndarray) -> np.ndarray:
        comming_refline_points = self.env.reference_line.get_reference_line_points(begin_idx=next_refline_idx,
                                                                end_idx= next_refline_idx + self.reference_line_points_lookahead * self.refeence_line_stride, 
                                                                extrapolate= True, 
                                                                stride = self.refeence_line_stride)
        
        assert comming_refline_points.shape[0] == self.reference_line_points_lookahead, f"Expected to get {self.reference_line_points_lookahead} points from reflinemanager, got {comming_refline_points.shape[0]}"
        
        comming_refline_points_rel_to_car : np.ndarray = np.array(car_orientation).dot((comming_refline_points - np.array(car_position)).T).T # (n,3)
        #comming_refline_points_rel_to_car_flattened = comming_refline_points_rel_to_car.ravel()
        return comming_refline_points_rel_to_car


    def get_mobil_states(self, game_states:SimStateData):
        # =======================================
        # Wheel States,Engine and Gearbox State
        # =======================================
        # gather all relevant information which describe the engine and the wheels states and pack them into a single array
        mobil:SceneVehicleCar = game_states.scene_mobil
        engine:Engine = mobil.engine
        gearbox_state = mobil.gearbox_state

        wheels: np.ndarray[SimulationWheel] = game_states.simulation_wheels
        wheels_states: list[RealTimeState] = [wheels[i].real_time_state for i in range(wheels.shape[0])]

        car_gear_and_wheels = np.array(
            [
                *(ws.is_sliding for ws in wheels_states),  # Bool (* is for unpacking)              size: 4
                *(ws.has_ground_contact for ws in wheels_states),  # Bool                           size: 4
                *(ws.damper_absorb for ws in wheels_states),  # 0.005 min, 0.15 max, 0.01 typically size: 4
                gearbox_state,  # Bool, except 2 at startup                                         size: 1
                engine.gear,  # 0 -> 5 approx                                                       size: 1
                engine.actual_rpm,  # 0-10000 approx                                                size: 1
                mobil.is_freewheeling, # Bool                                                       size: 1
                # linesight (from tminterface discord):
                #    n_contact_material_physics_behavior_types (here NUM_SURFACE_CATEGORIES) is the number of possible 
                #    materials a wheel can touch, we added this thinking it would help the agent understand the different
                #    behaviors of the car on different surfaces by having the info more than visually, again we are not sure 
                #    if this input is useful we did not do proper ablation tests because each test is very long.
                # TODO: Could we only store 4 values saying on which surface each wheel is instead of 4*NUM_SURFACE_CATEGORIES values
                *(
                    i == physics_behavior_fromint[ws.contact_material_id & 0xFFFF] # Doing & 0xFFFF masks the value, keeping only the lowest 16 bits and discarding any higher bits.
                    for ws in wheels_states
                    for i in range(NUM_SURFACE_CATEGORIES)
                ),                                                                              #    size: 4*NUM_SURFACE_CATEGORIES
            ],dtype=np.float32,)
        return car_gear_and_wheels
