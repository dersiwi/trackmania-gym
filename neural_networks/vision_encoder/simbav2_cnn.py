import math

import torch.nn as nn
from torch.nn.modules.conv import Conv2d
from torch.nn.common_types import _size_2_t

from neural_networks.vision_encoder.conv_NNs import VisionModel
from tmn_sb3.simbav2.simbav2_layers import HyperMLP


class UnitConv2D(Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype
        )
        # this enforces the weights to have unit norm
        self.weight._hyper_dense = True


class SimbaV2VisionModel(VisionModel):
    def __init__(self, out_dim: int, img_shape):
        encoder = nn.Sequential(
            UnitConv2D(img_shape[0], 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            UnitConv2D(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            UnitConv2D(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            UnitConv2D(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        def make_feature_projector(in_dim, out_dim):
            hidden_dim = 128
            return HyperMLP(
                in_features=in_dim,
                out_features=out_dim,
                hidden_features=hidden_dim,
                scaler_init=math.sqrt(2 / hidden_dim),
                scaler_scale=math.sqrt(2 / hidden_dim),
            )

            layers = [HyperMLP()]
            return nn.Sequential(*layers)

        super().__init__(
            encoder=encoder, feature_projector=lambda in_dim: make_feature_projector(in_dim, out_dim), img_shape=img_shape
        )
