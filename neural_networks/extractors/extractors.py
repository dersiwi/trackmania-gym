from __future__ import annotations
import math
import torch
import torch.nn as nn
import gymnasium as gym

from abc import ABC, abstractmethod
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Optional, List, Type, Dict, Any, Union
from dataclasses import dataclass, asdict
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import create_mlp

from neural_networks.extractors.utils import build_vision_model

from functools import partial

import torch.nn as nn
import hydra
from configs.config import TrainConfig, PolicyCfg
from utils.hydra_wandb_utils import secure_attribute_retrieval
from configs.config import ModelCfg

from tmn_sb3.simbav2.simbav2_layers import HyperMLP

@dataclass
class ExtractorConfig:
    """
    Args:
        observation_space: Gym Dict space describing the full observation structure.
        out_dim: The number of dimension each extractor should project on to
        normalized_image: If True, assumes that image inputs are already normalized.        
        float_model (list[int]): Optional list defining MLP layer sizes for vector inputs.
        activation_fn (type[nn.Module]): Activation function class for MLPs.
        last_activation_fn (type[nn.Module]): Activation for final MLP layer.
        check_channnles (bool): Whether to do or not the check for the number of channels.
            e.g., with frame-stacking, the observation space may have more channels than expected. 
    """
    # observation_space: gym.spaces.Space
    vision_model_kwargs: ModelCfg
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
    
    def create(policy_cfg: PolicyCfg, cfg: TrainConfig, vision_model_kwargs: Dict[str,Any], device: str,normalized_images:bool=False) -> ExtractorConfig:
        """Creates a base ExtractorConfig with all shared/common parameters."""
        policy_cfg = cfg.policy
        activation_fn_class = secure_attribute_retrieval(
            lambda: hydra.utils.get_class(policy_cfg.activation_fn._target_), nn.ReLU
        )
        last_activation_fn_class = secure_attribute_retrieval(
            lambda: hydra.utils.get_class(policy_cfg.last_activation_fn._target_), nn.Identity
        )
        
        return ExtractorConfig(
            vision_model_kwargs=vision_model_kwargs, 
            normalized_image=secure_attribute_retrieval(lambda: cfg.rl_env.env.normalize_images,normalized_images),
            out_dim=secure_attribute_retrieval(lambda: policy_cfg.extractors_out_dim, 64),
            check_channels=secure_attribute_retrieval(lambda: cfg.rl_env.obs_manager.check_channels,False),
            float_model=secure_attribute_retrieval(lambda: policy_cfg.float_net, None), 
            activation_fn=activation_fn_class,
            last_activation_fn=last_activation_fn_class,
            device = device 
        )


class TMN_Extractor(BaseFeaturesExtractor, ABC):
    """Feature extractor class from which special extractor can extend
    Args:
         observation_space: Gym Dict space describing the full observation structure.
         vision_model_kwargs (vision_model_kwargs) : Hydra loaded config file needed to instanciate the vision model
         out_dim: The number of dimension each extractor should project on to
         normalized_image: If True, assumes that image inputs are already normalized.        
         float_model (list[int]): Optional list defining MLP layer sizes for vector inputs.
         activation_fn (type[nn.Module]): Activation function class for MLPs.
         last_activation_fn (type[nn.Module]): Activation for final MLP layer.
         check_channnles (bool): Whether to do or not the check for the number of channels.
            e.g., with frame-stacking, the observation space may have more channels than expected. 
    """
    def __init__(self, observation_space: gym.spaces.Box,
            vision_model_kwargs: ModelCfg = None,
            out_dim: int = 64,
            device: str = "cpu",
            normalized_image: bool = False,
            float_model: Optional[List[int]] = None,
            activation_fn: Type[nn.Module] = nn.ReLU,
            last_activation_fn: Type[nn.Module] = nn.Tanh,
            check_channels: bool = True):

        super().__init__(observation_space, features_dim = out_dim)
        self.vision_model_kwargs = vision_model_kwargs
        self.out_dim = out_dim
        self.device = device
        self.normalized_image = normalized_image
        self.float_model = float_model
        self.activation_fn = activation_fn
        self.last_activation_fn = last_activation_fn
        self.check_channels = check_channels

    def build_box_extractor(self, space : gym.spaces.Space):
        if is_image_space(observation_space= space, check_channels=self.check_channels, normalized_image=self.normalized_image):
            if self.vision_model_kwargs is None:
                raise ValueError("vision_model_args must be provided for image spaces.")
            return build_vision_model(space=space, out_dim = self.out_dim, device = self.device, vision_model_kwargs=self.vision_model_kwargs)

        # Otherwise, handle vector (float) inputs
        input_dim = space.shape[0]
        hidden_dim = 128
        return HyperMLP(in_features=input_dim,out_features=self.out_dim,hidden_features=hidden_dim,scaler_init=math.sqrt(2/hidden_dim),scaler_scale=math.sqrt(2/hidden_dim))
        if self.float_model:
            layers = create_mlp(input_dim=input_dim, output_dim=self.out_dim, net_arch=self.float_model, activation_fn=self.activation_fn)
        else:
            hidden_dim = input_dim // 2 if input_dim > self.out_dim else input_dim * 2
            layers = [
                nn.Linear(input_dim, hidden_dim, device=self.device),
                self.activation_fn(),
                nn.Linear(hidden_dim, self.out_dim, device=self.device),
                self.last_activation_fn(),
            ]
        return nn.Sequential(*layers)
    
    @abstractmethod
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

class TMN_Box_Extractor(TMN_Extractor):
    """Feature extractor for a single gym.Box observation (image or vector)."""

    def __init__(self, observation_space : gym.spaces.Box, vision_model_kwargs = None, out_dim = 64, device = "cpu", normalized_image = False, float_model = None, activation_fn = nn.ReLU, last_activation_fn = nn.Tanh, check_channels = True):
        super().__init__(observation_space, vision_model_kwargs, out_dim, device, normalized_image, float_model, activation_fn, last_activation_fn, check_channels)
        assert isinstance(observation_space,gym.spaces.Box), f"This extractor only works with Box observation spaces but got {observation_space}"
        self.extractor = self.build_box_extractor(self._observation_space)


    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.extractor(observations)

class TMN_Dict_Extractor(TMN_Extractor):
    """Combined feature extractor for dictionary observations (images + vectors)."""

    def __init__(self, observation_space : gym.spaces.Dict, vision_model_kwargs = None, out_dim = 64, device = "cpu", normalized_image = False, float_model = None, activation_fn = nn.ReLU, last_activation_fn = nn.Tanh, check_channels = True):
        assert isinstance(observation_space,gym.spaces.Dict), f"This extractor only works with Dict observation spaces but got {observation_space}"
        super().__init__(observation_space, vision_model_kwargs, out_dim, device, normalized_image, float_model, activation_fn, last_activation_fn, check_channels)

        extractors = {}
        total_dim = 0
        for key, subspace in observation_space.spaces.items():
            extractor = self.build_box_extractor(subspace)
            extractors[key] = extractor
            total_dim += out_dim

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_dim

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = [extractor(observations[key]) for key, extractor in self.extractors.items()]
        return torch.cat(encoded, dim=1)
