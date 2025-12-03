from typing import Type, Optional, Any, Union
import gymnasium as gym
from torch import nn

from .extractors import TMN_Box_Extractor, TMN_Dict_Extractor


def make_tmn_extractor(
    observation_space: gym.Space,
    *,
    out_dim: int = 64,
    device: str = "cpu",
    **kwargs: Any,
) -> Union[TMN_Box_Extractor, TMN_Dict_Extractor]:
    """
    Factory function that automatically selects and initializes the right
    TMN feature extractor (Box or Dict) based on the observation space type.

    Args:
        observation_space: Gym observation space (Box or Dict supported).
        vision_model: Vision model class (for image inputs).
        out_dim: Output dimension for each extractor.
        device: Torch device string.
        **kwargs: Additional keyword arguments passed directly to the extractor
                  (e.g. normalized_image, float_model, activation_fn, etc.)

    Returns:
        An initialized instance of TMN_Box_Extractor or TMN_Dict_Extractor.
    """

    if isinstance(observation_space, gym.spaces.Box):
        return TMN_Box_Extractor(
            observation_space=observation_space,
            out_dim=out_dim,
            device=device,
            **kwargs,  # forward everything else
        )

    elif isinstance(observation_space, gym.spaces.Dict):
        return TMN_Dict_Extractor(
            observation_space=observation_space,
            out_dim=out_dim,
            device=device,
            **kwargs,  # forward everything else
        )

    else:
        raise TypeError(
            f"Unsupported observation space type: {type(observation_space).__name__}. "
            "Only gym.spaces.Box and gym.spaces.Dict are supported."
        )

