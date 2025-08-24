from __future__ import annotations
import traceback
import numpy as np
from gymnasium import spaces
from dataclasses import dataclass

from tminterface.structs import (
    SimStateData, 
    HmsDynaStateStruct,
    SceneVehicleCar,
    SimulationWheel,
    RealTimeState,
    Engine)

from trackmania_env.observations.observation_manager import ObservationManager
from trackmania_env.utils.contact_materials import get_normalized_surface_float
from trackmania_env.utils.constants import NormalizationFactors



from dataclasses import dataclass

from trackmania_env.utils.statevector_indexing import IOSVBase
class IndependentObservationManager(ObservationManager):
    """The idea behind this observation manager is that other maangers, 
    like e.g. NextPoint-obs contain map-specific observation like comming reference line points.
    This manager does not use any track-specific observations."""

    def __init__(self, colorspace, convert_torch, img_width, img_height, obs_have_img = True, img_dump_freq = 1000000, n_dump_imgs = 20, normalize_obs = False):
        super().__init__(colorspace, convert_torch, img_width, img_height, obs_have_img, img_dump_freq, n_dump_imgs, normalize_obs)

        self.mobile_states_dim = 4 + 3*4 + 4
        self.statevector_dim = self.mobile_states_dim + 1 #+1 because of speed

        self.idxs = IOSVBase(0)

    def get_observation_dict(self) -> spaces.Dict:
        """Returns observation dict for environment according to initialization."""
        return spaces.Dict({
                "image": spaces.Box(low=0, high=255, shape=(self.n_channels, self.img_height, self.img_width), dtype=np.uint8),
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(self.statevector_dim,), dtype=np.float32),
            })
    
    def normalize_state_vector(self, obs : np.ndarray):
        """Normalizes the state vector. This method implements regularization by deviding by the max value; due to booleans in the obsrvazations."""
        obs[self.idxs.speed_idx]            /= NormalizationFactors.speed_norm                  
        obs[self.idxs.gearbox_state_idx]    /= NormalizationFactors.gearbox_norm                      
        obs[self.idxs.actual_rpm_idx]       /= NormalizationFactors.rpm_norm                 
        obs[self.idxs.gear_idx]             /= NormalizationFactors.gear_norm
        return obs
    

    def get_values_from_state_dict(self, obs : SimStateData):

        mobile_states = self.get_mobil_states(obs)
        speed = obs.display_speed
    
        floatvec : np.ndarray = np.hstack([mobile_states, speed], dtype =  np.float32)
        assert floatvec.shape[0] == self.statevector_dim, f"Floatvector has size {floatvec.shape[0]}, however should be size {self.statevector_dim}"
        
        self.info["speed"] = speed
        self.info["mobile_states"] = mobile_states

        return floatvec

    def get_car_orientation(self, dyna_current : HmsDynaStateStruct) -> np.ndarray:
        """Extracts the cars current orientation in world coordinate system:
        Args:
            dyna_current (HmsDynaStateStruct) : You can get this from obs.dyna.current_state (if obs is a SimStateData-object.)"""
        orientation = np.array(dyna_current.rotation.to_numpy(), dtype=float).T
        return orientation


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
                *get_normalized_surface_float(wheels_states)                                       #size: 4
            ],dtype=np.float32,)
        return car_gear_and_wheels
