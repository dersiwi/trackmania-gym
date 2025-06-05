import numpy as np
from typing import Dict
import gymnasium as gym 
from gymnasium import ObservationWrapper
from gymnasium.vector import VectorObservationWrapper


class BGRA_to_RGB(ObservationWrapper):
    """This is a wrapper which converts the bgra image field into rgb image"""
    def __init__(self, env):
        super().__init__(env)
        assert 'image' in self.env.observation_space.spaces
        image = self.env.observation_space["image"]
        assert isinstance(image, gym.spaces.Box)
        assert (
            len(image.shape) == 3
            and image.shape[-1] == 4
        )
        assert (
            np.all(image.low == 0)
            and np.all(image.high == 255)
            and image.dtype == np.uint8
        )
        self.env.observation_space["image"] = gym.spaces.Box(0, 255, (image.shape[0], image.shape[1],3), np.uint8)
    
    def observation(self, observation : Dict[str,any]):
        assert "image" in observation
        observation["image"] = observation["image"][..., [2, 1, 0]]
        return observation


class Vec_BGRA_to_RGB(VectorObservationWrapper):
    """This is a wrapper which converts the bgra image field into rgb image"""
    def __init__(self, env):
        super().__init__(env)
        assert 'image' in self.env.observation_space.spaces
        image = self.env.observation_space["image"]
        assert isinstance(image, gym.spaces.Box)
        assert (
            len(image.shape) >= 3
            and image.shape[-1] == 4
        )
        assert (
            np.all(image.low == 0)
            and np.all(image.high == 255)
            and image.dtype == np.uint8
        )
        self.env.observation_space["image"] = gym.spaces.Box(0, 255, (image.shape[0], image.shape[1],image.shape[2],3), np.uint8)
    
    def observation(self, observation : Dict[str,any]):
        assert "image" in observation
        observation["image"] = observation["image"][..., [2, 1, 0]]
        return observation