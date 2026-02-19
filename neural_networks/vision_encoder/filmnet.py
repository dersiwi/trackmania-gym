from itertools import pairwise

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from neural_networks.vision_encoder.conv_NNs import VisionModel
from tmn_sb3.simbav2.simbav2_layers import HyperDense

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

    def forward(self, x: th.Tensor, gammas: th.Tensor, betas: th.Tensor):
        # x shape: (B, C, H, W)
        # gammas, betas shape: (B, C)
        gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        return (gammas * x) + betas


class FiLMedResBlock(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) residual block.

    This block implements the FiLM-conditioned residual architecture
    described in Section 2.2 and illustrated in Figure 3 (right)
    of the original FiLM paper.
    """

    def __init__(self, in_channels: int, out_channels: int, conv_class=nn.Conv2d):
        super().__init__()

        self.actual_in = in_channels + 2  # In dim  by 2 because we concatenate (x, y) coords

        self.conv1x1 = conv_class(self.actual_in, out_channels, kernel_size=1)
        # Use padding=1 to preserve spatial resolution (H, W).
        # The number of output channels must match `out_channels` from the conv1x1
        # so that the residual connection can work.
        self.conv3x3 = conv_class(out_channels, out_channels, kernel_size=3, padding=1)

        self.bn = nn.Identity() if issubclass(conv_class, UnitConv2D) else nn.BatchNorm2d(out_channels, affine=False)

        self.film = FiLM()

    def forward(self, x, gamma, beta):
        """
        x: (B, C, H, W)
        gamma: (B, out_channels)
        beta: (B, out_channels)
        """
        x_with_coords = add_coords(x)  # shape (B,C +2, H,W)
        out = self.conv1x1(x_with_coords)  # shape (B,out_channels, H,W)
        out = F.relu(out)
        identity = out

        out = self.conv3x3(out)
        out = self.bn(out)
        # for every feature map/ channel we need a gamma and beta
        assert gamma.shape[-1] == beta.shape[-1] == out.shape[1]
        out = self.film(out, gamma, beta)
        out = F.relu(out)

        # Skip Connection after the final ReLU
        assert out.shape == identity.shape
        return out + identity


class FiLMedNetwork(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) network but without final linear layer.

    This block implements the FiLM-conditioned network described in Section 2.2
    and illustrated in Figure 3 (middle) of the original FiLM paper.
    """

    def __init__(
        self,
        in_channels: int,
        feature_dim_per_block: list[int],
        output_dim: int = 512,
        conv_class=nn.Conv2d,
    ):
        super().__init__()
        self.feature_dim_per_block = feature_dim_per_block

        # We create a list of dims starting with the input image channels
        all_dims = [in_channels, *feature_dim_per_block]

        # creates FiLMedResBlocks with in_d out_d pairs
        self.layers = nn.ModuleList([FiLMedResBlock(in_d, out_d, conv_class=conv_class) for in_d, out_d in pairwise(all_dims)])

        # Final projection before pooling, refer to section 2.2
        self.final_conv = conv_class(feature_dim_per_block[-1], output_dim, kernel_size=1)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x: th.Tensor, gammas: th.Tensor | None = None, betas: th.Tensor | None = None):
        """
        x: (B, C, H, W) raw image
        gammas/betas: List of (B, out_dim) tensors, length == num_blocks
        """
        # NOTE: this should only trigger when the model gets callled with a dummy input for initialisation
        if gammas is None or betas is None:
            gammas = [th.ones(x.size(0), dim, device=x.device) for dim in self.feature_dim_per_block]
            betas = [th.zeros(x.size(0), dim, device=x.device) for dim in self.feature_dim_per_block]
 
        for i, layer in enumerate(self.layers):
            x = layer(x, gammas[i], betas[i])
        # Final head
        x = self.final_conv(x)
        x = self.pool(x)
        x = add_coords(x)
        return self.flatten(x)


class FiLMedVisionModel(VisionModel):
    """
    Feature-wise Linear Modulation (FiLM) network but WITH THE FINAL LINEAR LAYER.

    This block implements the FiLM-conditioned network described in Section 2.2
    and illustrated in Figure 3 (middle) of the original FiLM paper.
    """

    def __init__(
        self,
        out_dim: int,
        img_shape: tuple,
        feature_dim_per_block: list[int],
        film_net_out_dim_before_linear: int = 512,
        final_linear_hidden_dim: int = 1024,
        conv_class=nn.Conv2d,
    ):
        encoder = FiLMedNetwork(
            in_channels=img_shape[0],
            feature_dim_per_block=feature_dim_per_block,
            output_dim=film_net_out_dim_before_linear,
            conv_class=conv_class,
        )

        linear_class = HyperDense if issubclass(conv_class, UnitConv2D) else nn.Linear

        # since we are not doing classification we are not using the softmax at the end
        def make_feature_projector(in_dim):
            return nn.Sequential(
                linear_class(in_dim, final_linear_hidden_dim), nn.ReLU(), linear_class(final_linear_hidden_dim, out_dim)
            )

        super().__init__(encoder=encoder, feature_projector=make_feature_projector, img_shape=img_shape)

    def forward(self, x: th.Tensor, gammas: th.Tensor, betas: th.Tensor):
        """
        Processes the image through the FiLMed blocks and then the MLP head.
        """
        x = self.encoder(x, gammas, betas)
        x = self.feature_projector(x)
        return x


class FiLMGenerator(nn.Module):
    """
    The FiLM generator which produces the beta and gammas for the film entwokd figure 3 left and section 2.2, here 
    feature_dim_per_block are the number of channel of the featur map for each resblock 
    """

    def __init__(self, in_dim, feature_dim_per_block: list[int], linear_class=nn.Linear, hidden_dim:int=512):
        super().__init__()

        self.feature_dim_per_block = feature_dim_per_block
        # for every res_block and its feature map we create num_channels * 2(one gamma, one beta) weights 
        self.total_output_size = 2 * sum(self.feature_dim_per_block)

        self.mlp = nn.Sequential(linear_class(in_dim, hidden_dim), nn.ReLU(), linear_class(hidden_dim, self.total_output_size))

    def forward(self, x):
        # (Batch, total_output_size)
        flat_output = self.mlp(x)
        
        gammas = []
        betas = []
        pointer = 0 # Tracks where we are in the flat_output

        # Slice the output based on the specific size of each block
        for channels in self.feature_dim_per_block:
            # Each block takes a slice of size (channels * 2)
            # Layout for one block: [gamma_0...gamma_N, beta_0...beta_N]
            block_params = flat_output[:, pointer : pointer + (2 * channels)]
            
            # Split the slice into gamma and beta
            g = block_params[:, :channels]
            b = block_params[:, channels:]

            # Apply the identity shift (gamma_baseline = 1.0)
            g = g + 1.0 # dunno saw this somewhere in the code
            
            gammas.append(g)
            betas.append(b)
            
            # Move the pointer to the start of the next block's parameters
            pointer += (2 * channels)

        assert gammas is not None and betas is not None
        return gammas, betas 
