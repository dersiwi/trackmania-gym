import numpy as np
from typing import Dict
import gymnasium as gym 
from gymnasium import ObservationWrapper

from gymnasium.wrappers.vector import GrayscaleObservation

WEIGHTS = np.array([0.2125, 0.7154, 0.0721])
class TransformGrayscale(ObservationWrapper):
    """This is a wrapper which converts the rgb image field into grayscale"""
    def __init__(self, env, keep_dim: bool = False):
        super().__init__(env)
        assert "image" in self.env.observation_space.spaces
        image = self.env.observation_space["image"]
        assert isinstance(image, gym.spaces.Box)
        assert (
            len(image.shape) == 3
            and image.shape[-1] == 3
        )
        assert (
            np.all(image.low == 0)
            and np.all(image.high == 255)
            and image.dtype == np.uint8
        )
        shape = (image.shape[0], image.shape[1],1) if keep_dim else (image.shape[0], image.shape[1])
        self.env.observation_space["image"] = gym.spaces.Box(0, 255, shape, np.uint8)
        if keep_dim:
            self.transform = lambda obs: np.expand_dims(
            np.sum(np.multiply(obs, WEIGHTS), axis=-1).astype(np.uint8),
            axis=-1)
        else:
            self.transform = lambda obs: np.sum(np.multiply(obs, WEIGHTS), axis=-1).astype(np.uint8)

    def observation(self, observation : Dict[str,any]):
        assert "image" in observation
        observation["image"] = self.transform(observation["image"])
        return observation
