from typing import Union
import numpy as np
from gymnasium.spaces import Box
from tminterface.structs import SimStateData, RealTimeState, SimulationWheel, SceneVehicleCar, Engine

from trackmania_env.observations.observation_term import ObservationTerm, VectorlikeTerm
from trackmania_env.utils.constants import MAX_SPEED, ObsNormalizationFactors
from trackmania_env.utils.contact_materials import physics_behavior_fromint, NUM_SURFACE_CATEGORIES
from game_interaction.ipc_fields import IPCFields


class SpeedTerm(VectorlikeTerm):
    def __init__(self, name: str = "speed", normalize: bool = True):
        super().__init__(name, normalize, dimension=1)
        self.observation_space = Box(
            low=0.0,
            high=1.0 if normalize else MAX_SPEED,
            shape=(),
            dtype=np.float32
        )

    def _normalize(self, obs: float) -> float:
        return obs / ObsNormalizationFactors.speed_norm

    def _get_obs(self, game_states: dict[str, Union[np.ndarray, SimStateData]], **kwargs) -> np.ndarray:
        return np.array([game_states[IPCFields.SIMSTATE].display_speed], dtype=np.float32), {}


class SurfaceFloats(VectorlikeTerm):
    """
    Returns a float vector representing the surface category each wheel is currently in contact with.
    """
    def __init__(self, name: str = "surface_floats", normalize: bool = True):
        super().__init__(name, normalize, 4)
        self.observation_space = Box(
            low=0.0,
            high=float(NUM_SURFACE_CATEGORIES + 1),
            shape=(4,),  # one float per wheel
            dtype=np.float32
        )

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        return obs  # already normalized 

    def _get_obs(self, game_states: dict[str, Union[np.ndarray, SimStateData]]) -> np.ndarray:
        """This method just transforms this one-hot-encoded vector [1. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0. 1. 0. 0. 0.] into [0.2 0.2 0.2 0.2]"""
        ssd : SimStateData = game_states[IPCFields.SIMSTATE]
        wheels_states: list[RealTimeState] = [wheel.real_time_state for wheel in ssd.simulation_wheels]
        surface_floats = []

        for ws in wheels_states:
            material_id = ws.contact_material_id & 0xFFFF
            matched_indices = [
                i for i in range(NUM_SURFACE_CATEGORIES)
                if i == physics_behavior_fromint[material_id]
            ]
            if not matched_indices:
                surface_floats.append(0.0)
            else:
                assert len(matched_indices) == 1, matched_indices
                idx = matched_indices[0]
                value = idx + 1 / (NUM_SURFACE_CATEGORIES + 1)  # +1 because sometimes its just no surface category (wheel is in the air.)
                surface_floats.append(float(value))

        return np.array(surface_floats, dtype=np.float32), {}

class MobileStatesTerm(VectorlikeTerm):
    """Returns all relevant dynamic car states"""

    def __init__(self, name = "mobile states", normalize=True):
        super().__init__(name,normalize, 16)
        
        # Define lower and upper bounds for each of the 16 features
        low = np.array(
            [0.0] * 8 +         # is_sliding and has_ground_contact (bools)
            [0.0] * 4 +         # damper_absorb
            [0.0, 0.0, 0.0, 0.0],  # gearbox_state, gear, rpm, is_freewheeling
            dtype=np.float32
        )

        high = np.array(
            [1.0] * 8 +         # is_sliding and has_ground_contact
            [0.2] * 4 +         # damper_absorb: typical range ~0.005 to 0.15, max ~0.2
            [2.0, 6.0, 10000.0, 1.0],  # gearbox_state, gear, rpm, is_freewheeling
            dtype=np.float32
        )

        self.observation_space = Box(low=low, high=high, shape=(16,), dtype=np.float32)

    def _normalize(self, obs):
        obs[12]             /= ObsNormalizationFactors.gearbox_norm
        obs[13]             /= ObsNormalizationFactors.gear_norm
        obs[14]             /= ObsNormalizationFactors.rpm_norm
        return obs
    
    def _get_obs(self, game_states: dict[str, Union[np.ndarray, SimStateData]]) -> np.ndarray:
        """Wheel States,Engine and Gearbox State
        gather all relevant information which describe the engine and the wheels states and pack them into a single array"""
        ssd : SimStateData = game_states[IPCFields.SIMSTATE]
        
        mobil:SceneVehicleCar = ssd.scene_mobil
        engine:Engine = mobil.engine
        gearbox_state = mobil.gearbox_state

        wheels: np.ndarray[SimulationWheel] = ssd.simulation_wheels
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

        return car_gear_and_wheels, {}