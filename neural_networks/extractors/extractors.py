from .utils import build_box_extractor
from typing import Optional, List, Type, Dict, Any, Union
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from dataclasses import dataclass, asdict


@dataclass
class ExtractorConfig:
    """   
    observation_space: Gym Dict space describing the full observation structure.
    vision_model: A neural network used to extract features from image observations.
    out_dim: The number of dimension each extractor should project on to
    normalized_image: If True, assumes that image inputs are already normalized.        
    float_model (list[int]): Optional list defining MLP layer sizes for vector inputs.
    activation_fn (type[nn.Module]): Activation function class for MLPs.
    last_activation_fn (type[nn.Module]): Activation for final MLP layer.
    check_channnles (bool): Whether to do or not the check for the number of channels.
        e.g., with frame-stacking, the observation space may have more channels than expected. 
    """
    # observation_space: gym.spaces.Space
    vision_model: Union[nn.Module, Any]
    vision_model_kwargs: Optional[Dict[str, Any]]
    out_dim: int
    normalized_image: bool
    float_model: Optional[List[int]]
    activation_fn: Type[nn.Module]
    last_activation_fn: Type[nn.Module]
    check_channels: bool
    device: str

    def to_dict(self, exclude_none: bool = True) -> Dict[str, Any]:
        """Return all configuration parameters as a dict suitable for `**kwargs` unpacking."""
        d = asdict(self)
        if exclude_none:
            return {k: v for k, v in d.items() if v is not None}
        return d

class TMN_Box_Extractor(BaseFeaturesExtractor):
    """Feature extractor for a single gym.Box observation (image or vector).

    :param observation_space: Gym Dict space describing the full observation structure.
    :param vision_model: A neural network used to extract features from image observations.
    :param out_dim: The number of dimension each extractor should project on to
    :param normalized_image: If True, assumes that image inputs are already normalized.        
    :param float_model (list[int]): Optional list defining MLP layer sizes for vector inputs.
    :param activation_fn (type[nn.Module]): Activation function class for MLPs.
    :param last_activation_fn (type[nn.Module]): Activation for final MLP layer.
    :param check_channnles (bool): Whether to do or not the check for the number of channels.
        e.g., with frame-stacking, the observation space may have more channels than expected. 
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        vision_model: Optional[Type[nn.Module]] = None,
        vision_model_kwargs: Optional[Dict[str, Any]] = None,
        out_dim: int = 64,
        device: str = "cpu",
        normalized_image: bool = False,
        float_model: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        last_activation_fn: Type[nn.Module] = nn.Tanh,
        check_channels: bool = True, 
    ) -> None:
        assert isinstance(observation_space,gym.spaces.Box), f"This extractor only works with Box observation spaces but got {observation_space}"
        super().__init__(observation_space, features_dim=out_dim)

        self.extractor = build_box_extractor(
            space =observation_space,
            out_dim=out_dim,
            device=device,
            vision_model_cls=vision_model,
            vision_model_kwargs= vision_model_kwargs,
            float_model=float_model,
            activation_fn=activation_fn,
            last_activation_fn=last_activation_fn,
            normalized_image=normalized_image,
            check_channels=check_channels,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.extractor(observations)

class TMN_Dict_Extractor(BaseFeaturesExtractor):
    """Combined feature extractor for dictionary observations (images + vectors).
    
    :param observation_space: Gym Dict space describing the full observation structure.
    :param vision_model: A neural network used to extract features from image observations.
    :param out_dim: The number of dimension each extractor should project on to
    :param normalized_image: If True, assumes that image inputs are already normalized.        
    :param float_model (list[int]): Optional list defining MLP layer sizes for vector inputs.
    :param activation_fn (type[nn.Module]): Activation function class for MLPs.
    :param last_activation_fn (type[nn.Module]): Activation for final MLP layer.
    :param check_channnles (bool): Whether to do or not the check for the number of channels.
        e.g., with frame-stacking, the observation space may have more channels than expected. 
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        vision_model: Optional[Type[nn.Module]] = None,
        vision_model_kwargs: Optional[Dict[str, Any]] = None,
        out_dim: int = 64,
        device: str = "cpu",
        normalized_image: bool = False,
        float_model: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        last_activation_fn: Type[nn.Module] = nn.Tanh,
        check_channels: bool = True, 
    ) -> None:
        assert isinstance(observation_space,gym.spaces.Dict), f"This extractor only works with Dict observation spaces but got {observation_space}"
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_dim = 0
        for key, subspace in observation_space.spaces.items():
            extractor = build_box_extractor(
                space=subspace,
                out_dim=out_dim,
                device=device,
                vision_model_cls=vision_model,
                vision_model_kwargs= vision_model_kwargs,
                float_model=float_model,
                activation_fn=activation_fn,
                last_activation_fn=last_activation_fn,
                normalized_image=normalized_image,
                check_channels=check_channels,
            )
            extractors[key] = extractor
            total_dim += out_dim

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_dim

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = [extractor(observations[key]) for key, extractor in self.extractors.items()]
        return torch.cat(encoded, dim=1)
