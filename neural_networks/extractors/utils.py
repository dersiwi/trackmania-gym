from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import gymnasium as gym

import functools

def build_vision_model(
    vision_model_cls: functools.partial[nn.Module],
    space: gym.Space,
    out_dim: int,
    device: str,
    vision_model_kwargs: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Instantiate and validate a vision model."""
    if not isinstance(vision_model_cls, functools.partial):
        raise TypeError("vision_model_cls must be functools.partial instance")
    
    vision_model_kwargs = vision_model_kwargs if vision_model_kwargs else {}
    vision_model_kwargs.setdefault("img_shape", space.shape)
    vision_model_kwargs.setdefault("out_dim", out_dim)
    vision_model_kwargs.pop("args", None)
    model = vision_model_cls(**vision_model_kwargs).to(device)

    dummy_input = torch.zeros(1, *space.shape, device=device)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    if dummy_output.ndim != 2:
        raise ValueError(f"Vision model output must have shape (N, D), got {dummy_output.shape}")
    if dummy_output.shape[1] != out_dim:
        raise ValueError(f"Vision model output dimension {dummy_output.shape[1]} != expected {out_dim}")
    return model