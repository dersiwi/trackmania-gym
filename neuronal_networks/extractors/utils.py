
from typing import Optional, List, Type
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import create_mlp

def build_vision_model(
    vision_model_cls: Type[nn.Module],
    space: gym.Space,
    out_dim: int,
    device: str,
) -> nn.Module:
    """Instantiate and validate a vision model."""
    if not (isinstance(vision_model_cls, type) and issubclass(vision_model_cls, nn.Module)):
        raise TypeError("vision_model must be a PyTorch nn.Module class, not an instance.")

    model = vision_model_cls().to(device)

    dummy_input = torch.zeros(1, *space.shape, device=device)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    if dummy_output.ndim != 2:
        raise ValueError(f"Vision model output must have shape (N, D), got {dummy_output.shape}")
    if dummy_output.shape[1] != out_dim:
        raise ValueError(f"Vision model output dimension {dummy_output.shape[1]} != expected {out_dim}")
    return model

def build_box_extractor(
    space: gym.Space,
    out_dim: int,
    device: str,
    vision_model_cls: Optional[Type[nn.Module]] = None,
    float_model: Optional[List[int]] = None,
    activation_fn: Type[nn.Module] = nn.ReLU,
    last_activation_fn: Type[nn.Module] = nn.Tanh,
    normalized_image: bool = False,
    check_channels: bool = True, 
) -> nn.Module:
    """
    Builds a feature extractor for a single gym.Box space.
    Uses vision_model_cls if the space is an image; otherwise builds an MLP.
    """
    if is_image_space(observation_space= space, check_channels=check_channels, normalized_image=normalized_image):
        if vision_model_cls is None:
            raise ValueError("vision_model_cls must be provided for image spaces.")
        return build_vision_model(vision_model_cls, space, out_dim, device)

    # Otherwise, handle vector (float) inputs
    input_dim = space.shape[0]
    if float_model:
        layers = create_mlp(
            input_dim=input_dim,
            output_dim=out_dim,
            net_arch=float_model,
            activation_fn=activation_fn,
            output_activation=last_activation_fn,
        )
    else:
        hidden_dim = input_dim // 2 if input_dim > out_dim else input_dim * 2
        layers = [
            nn.Linear(input_dim, hidden_dim, device=device),
            activation_fn(),
            nn.Linear(hidden_dim, out_dim, device=device),
            last_activation_fn(),
        ]
    return nn.Sequential(*layers)
