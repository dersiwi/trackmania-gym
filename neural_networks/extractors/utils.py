from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import gymnasium as gym
import hydra

from configs.config import ModelCfg

def build_vision_model(space: gym.Space, out_dim: int, device: str, vision_model_kwargs: ModelCfg = None,) -> nn.Module:
    # 
    """Instantiate and validate a vision model. This method basically uses everytrhing specified in the [visionmodelname].yaml and uses it
    to instanciate the vision model.
    Args:
        space (gym.Space)   : Used for image-dimensions in order to test a dummy-output
        out_dim (int)       : Specifies the size of the last layer of the vision model
        device (str)        : Device on which the vision model is stored
        vision_model_kwargs (ModelCfg)  : Basically the loaded yaml-file (loaded by hydra.)
    Returns:
        Instanciated Vision Model as specified in the ModelCfg."""

    vision_model : nn.Module = hydra.utils.instantiate(vision_model_kwargs, **{"img_shape" : space.shape, "out_dim" : out_dim})
    vision_model.to(device)

    dummy_input = torch.zeros(1, *space.shape, device=device)
    with torch.no_grad():
        dummy_output = vision_model(dummy_input)

    if dummy_output.ndim != 2:
        raise ValueError(f"Vision model output must have shape (N, D), got {dummy_output.shape}")
    if dummy_output.shape[1] != out_dim:
        raise ValueError(f"Vision model output dimension {dummy_output.shape[1]} != expected {out_dim}")
    return vision_model