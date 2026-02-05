from .simbav2_layers import (
    HyperCategoricalValue,
    HyperEmbedder,
    HyperLERPBlock,
    HyperNormalTanhPolicy,
)

import torch
import torch.nn as nn


class SimbaV2Critic(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        in_features: int,
        hidden_features: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        c_shift: float,
        num_bins: int,
        min_v: float,
        max_v: float,
        gain: float = 1.0,
    ) -> None:
        super().__init__()

        # TODO: in_features must be dimension of the action concatenated with the observations at axis 1
        # this means that the obs can not be dicts
        # maybe parse them observation spaces in the constructor and then calculated them manually
        self.embedder = HyperEmbedder(
            in_features=in_features,
            out_features=hidden_features,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            c_shift=c_shift,
            gain=gain,
        )

        lerp_blocks = [
            HyperLERPBlock(
                in_features=hidden_features,
                out_features=hidden_features,
                hidden_features=hidden_features,
                scaler_init=scaler_init,
                scaler_scale=scaler_scale,
                alpha_init=alpha_init,
                alpha_scale=alpha_scale,
                gain=gain,
            )
            for _ in range(num_blocks)
        ]

        self.encoder = nn.Sequential(*lerp_blocks)

        self.predictor = HyperCategoricalValue(
            in_features=hidden_features,
            hidden_features=hidden_features,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor):
        x = torch.concatenate((observations, actions), dim=-1)  # original code uses dim = 1
        x = self.embedder(x)
        x = self.encoder(x)
        q = self.predictor(x)
        return q
