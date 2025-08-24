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
from trackmania_env.utils.contact_materials import physics_behavior_fromint,NUM_SURFACE_CATEGORIES
from trackmania_env.utils.constants import ObsNormalizationFactors
from numba import jit


def get_wheel_states(game_states : SimStateData) -> tuple[list[RealTimeState], np.ndarray[SimulationWheel]]:
    wheels: np.ndarray[SimulationWheel] = game_states.simulation_wheels
    wheels_states : list[RealTimeState] = [wheels[i].real_time_state for i in range(wheels.shape[0])]
    return wheels_states, wheels

class ObservationTerm:
    """This class contains observation terms for vecotr-like observations.
    Each observation-term returns a numpy-array of shape [N,]."""

    def __init__(self, dim : int, normalize : bool):
        self.dim = dim
        self.normalize = normalize
    
    def _get_obs(self, gamestates : SimStateData, **kwargs) -> np.ndarray:
        raise NotImplementedError()
    
    def _normalize(self, obs) -> np.ndarray:
        raise NotImplementedError()

    def get_observation(self, game_states : SimStateData, **kwargs) -> np.ndarray:
        obs = self._get_obs(game_states, **kwargs)
        if self.normalize:
            obs = self._normalize(obs)
        return obs
    

class SurfaceFloats(ObservationTerm):
    

    def __init__(self, normalize):
        super().__init__(4, normalize)

    def _normalize(self, obs):
        return obs # is always normalized

    def _get_obs(self, game_states : SimStateData):
        """# linesight (from tminterface discord): TODO : Big refactor!!
            #    n_contact_material_physics_behavior_types (here NUM_SURFACE_CATEGORIES) is the number of possible 
            #    materials a wheel can touch, we added this thinking it would help the agent understand the different
            #    behaviors of the car on different surfaces by having the info more than visually, again we are not sure 
            #    if this input is useful we did not do proper ablation tests because each test is very long.
            # 
        This method just transforms this one-hot-encoded vector [1. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0.] into [0.2 0.2 0.2 0.2]"""
        
        wheels_states, wheels = get_wheel_states(game_states)
        wsm = []
        for ws in wheels_states:
            # Doing & 0xFFFF masks the value, keeping only the lowest 16 bits and discarding any higher bits.
            scidx = np.nonzero([i == physics_behavior_fromint[ws.contact_material_id & 0xFFFF] for i in range(NUM_SURFACE_CATEGORIES)])[0]
            if len(scidx) == 0: wsm.append(0)
            else:
                assert len(scidx)  == 1, scidx
                wsm.append(scidx[0] + 1 / (NUM_SURFACE_CATEGORIES + 1)) # +1 because sometimes its just no surface category (wheel is in the air.)
        return np.array(wsm, dtype = np.float32)

class MobileStatesTerm(ObservationTerm):
    """Returns all relevant dynamic car states"""

    def __init__(self, normalize):
        super().__init__(16, normalize)

    def _normalize(self, obs):
        obs[12]             /= ObsNormalizationFactors.gearbox_norm
        obs[13]             /= ObsNormalizationFactors.gear_norm
        obs[14]             /= ObsNormalizationFactors.rpm_norm
        return obs
    
    def _get_obs(self, game_states : SimStateData) -> np.ndarray:
        """Wheel States,Engine and Gearbox State
        gather all relevant information which describe the engine and the wheels states and pack them into a single array"""
        mobil:SceneVehicleCar = game_states.scene_mobil
        engine:Engine = mobil.engine
        gearbox_state = mobil.gearbox_state

        wheels: np.ndarray[SimulationWheel] = game_states.simulation_wheels
        wheels_states: list[RealTimeState] = [wheels[i].real_time_state for i in range(wheels.shape[0])]

        car_gear_and_wheels = np.array(
            [
                *(ws.is_sliding for ws in wheels_states),  # Bool (* is for unpacking)              size: 4 0:4
                *(ws.has_ground_contact for ws in wheels_states),  # Bool                           size: 4 4:8
                *(ws.damper_absorb for ws in wheels_states),  # 0.005 min, 0.15 max, 0.01 typically size: 4 8:12
                gearbox_state,  # Bool, except 2 at startup                                         size: 1 12
                engine.gear,  # 0 -> 5 approx                                                       size: 1 13
                engine.actual_rpm,  # 0-10000 approx                                                size: 1 14
                mobil.is_freewheeling, # Bool                                                       size: 1 15
            ],dtype=np.float32,)

        return car_gear_and_wheels

class SpeedTerm(ObservationTerm):
    def __init__(self, normalize):
        super().__init__(1, normalize)

    def _normalize(self, obs):
        return obs / ObsNormalizationFactors.speed_norm

    def _get_obs(self, gamestates, **kwargs):
        return gamestates.display_speed

class NextReflinePoint(ObservationTerm):
    def __init__(self, n_refline_points : int, normalize):
        super().__init__(n_refline_points * 3, normalize)
        self.n_refline_points = n_refline_points

    def _normalize(self, obs):
        return obs / ObsNormalizationFactors.refline_norm
    
    def _get_obs(self, gamestates, **kwargs):
        dyna_current: HmsDynaStateStruct = gamestates.dyna.current_state
        comming_refline_points : np.ndarray = kwargs["reflinepoints"]

        car_position = gamestates.position
        car_orientation = np.array(dyna_current.rotation.to_numpy(), dtype=float).T

        assert comming_refline_points.shape[0] == self.n_refline_points, f"Expected to get {self.n_refline_points} points from reflinemanager, got {comming_refline_points.shape[0]}"
        
        comming_refline_points_rel_to_car : np.ndarray = np.array(car_orientation).dot((comming_refline_points - np.array(car_position)).T).T # (n,3)
        #comming_refline_points_rel_to_car_flattened = comming_refline_points_rel_to_car.ravel()
        return comming_refline_points_rel_to_car.ravel()
    
class LateralDistance(ObservationTerm):

    def __init__(self, normalize):
        super().__init__(1, normalize)
    
    def _normalize(self, obs):
        return obs / ObsNormalizationFactors.lateral_dist_norm

    def _get_obs(self, gamestates, **kwargs):
        lateral_distance : np.ndarray = kwargs["lateral_distance"]
        return lateral_distance