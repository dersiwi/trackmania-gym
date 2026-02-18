from inspect import isclass
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.conv import Conv2d
from torch.nn.common_types import _size_2_t

from neural_networks.vision_encoder.conv_NNs import VisionModel
from tmn_sb3.simbav2.simbav2_layers import HyperMLP, HyperDense

from .simbav2_cnn import UnitConv2D


def add_coords(x: th.Tensor):
    """
    Concatenates relative (x, y) coordinates scaled from -1 to 1
    to the input tensor of shape (B, C, H, W).
    """
    batch_size, _, h, w = x.size()
    # Create coordinate grids
    y_coords = th.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(batch_size, 1, h, w)
    x_coords = th.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(batch_size, 1, h, w)
    # Concatenate along the channel dimension
    return th.cat([x, y_coords, x_coords], dim=1)


# ripped from https://github.com/ethanjperez/film/blob/master/vr/models/filmed_net.py#L18
class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """

    def forward(self, x, gammas, betas):
        # x shape: (B, C, H, W)
        # gammas, betas shape: (B, C)
        gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        return (gammas * x) + betas


# the FiLM-ed ResBlock from section 2.2
class FiLMedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_class=nn.Conv2d):
        super().__init__()
        # In dim  by 2 because we concatenate (x, y) coords
        actual_in = in_channels + 2

        self.conv1x1 = conv_class(actual_in, out_channels, kernel_size=1)

        self.conv3x3 = conv_class(out_channels, out_channels, kernel_size=3, padding=1)

        # for Batch Norm affine is False because FiLM provides the scale/shift
        self.bn = nn.Identity() if issubclass(conv_class, UnitConv2D) else nn.BatchNorm2d(out_channels, affine=False)

        self.film = FiLM()

        # Skip connection projection if dimensions don't match
        self.shortcut = nn.Identity()
        if actual_in != out_channels:
            self.shortcut = conv_class(actual_in, out_channels, kernel_size=1)

    def forward(self, x, gamma, beta):
        x_with_coords = add_coords(x)
        identity = self.shortcut(x_with_coords)

        out = self.conv1x1(x_with_coords)
        out = F.relu(out)

        out = self.conv3x3(out)
        out = self.bn(out)
        out = self.film(out, gamma, beta)
        out = F.relu(out)

        # Skip Connection after the final ReLU
        return out + identity


class FiLMedSimbaEncoder(nn.Module):
    def __init__(self, in_channels, feature_dim=128, output_dim=512, num_blocks=4, conv_class=nn.Conv2d):
        super().__init__()
        self.num_blocks = num_blocks
        self.feature_dim = feature_dim

        # Initial stem to get to internal feature_dim
        self.stem = conv_class(in_channels, feature_dim, kernel_size=3, padding=1)

        self.layers = nn.ModuleList(
            [FiLMedResBlock(feature_dim, feature_dim, conv_class=conv_class) for _ in range(num_blocks)]
        )

        # only feature_dim+2 if use we add cords
        self.final_conv = conv_class(feature_dim, output_dim, kernel_size=1)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x, gammas=None, betas=None):
        """
        x: (B, C, H, W)
        gammas: List of (B, feature_dim) tensors of length num_blocks
        betas: List of (B, feature_dim) tensors of length num_blocks
        """
        # Handle dummy pass for initialization or missing conditioning
        if gammas is None or betas is None:
            gammas = [th.ones(x.size(0), self.feature_dim, device=x.device) for _ in range(self.num_blocks)]
            betas = [th.zeros(x.size(0), self.feature_dim, device=x.device) for _ in range(self.num_blocks)]

        # Initial Stem
        x = F.relu(self.stem(x))

        # Loop through the dynamic number of blocks
        for i, layer in enumerate(self.layers):
            x = layer(x, gammas[i], betas[i])

        # Classifier input gets coordinates per paper specs
        # x = add_coords(x)
        x = self.final_conv(x)
        x = self.pool(x)
        return self.flatten(x)


class FiLMedVisionModel(VisionModel):
    def __init__(self, out_dim: int, img_shape, conv_class=nn.Conv2d):
        encoder = FiLMedSimbaEncoder(img_shape[0], conv_class=conv_class)

        def make_feature_projector(in_dim, out_dim):
            # Two-layer MLP with 1024 hidden units per paper section 2.2
            return nn.Sequential(nn.Linear(in_dim, 1024), nn.ReLU(), nn.Linear(1024, out_dim))

        super().__init__(
            encoder=encoder, feature_projector=lambda in_dim: make_feature_projector(in_dim, out_dim), img_shape=img_shape
        )


class SimpleFiLMGen(nn.Module):
    def __init__(self, in_dim, num_blocks, feature_dim, linear_class=nn.Linear, hidden_dim=512):
        super().__init__()
        self.num_blocks = num_blocks
        self.feature_dim = feature_dim

        # In the paper code: cond_feat_size = 2 * feature_dim (one gamma, one beta)
        self.total_output_size = num_blocks * (2 * feature_dim)

        self.mlp = nn.Sequential(
            linear_class(in_dim, hidden_dim), nn.ReLU(), linear_class(hidden_dim, self.total_output_size)
        )

    def forward(self, x):
        # Shape: (Batch, num_blocks * 2 * feature_dim)
        output = self.mlp(x)
        # Reshape to (Batch, num_blocks, 2, feature_dim)
        output = output.view(-1, self.num_blocks, 2, self.feature_dim)

        gammas = []
        betas = []

        for i in range(self.num_blocks):
            # see https://github.com/ethanjperez/film/blob/master/vr/models/film_gen.py#L160
            # The paper code does: out + gamma_baseline (usually 1.0)
            # This ensures that if the MLP outputs 0, gamma becomes 1 (no change to features)
            #TODO: is this correct 
            g = output[:, i, 0, :] + 1.0
            b = output[:, i, 1, :]

            gammas.append(g)
            betas.append(b)

        return gammas, betas
