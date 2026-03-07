from .simbav2_layers import HyperCategoricalValue, HyperEmbedder, HyperLERPBlock, HyperNormalTanhPolicy, HyperCategorialPolicy

import torch
import torch.nn as nn


def create_simbav2_base(
    num_blocks: int,
    in_features: int,
    hidden_features: int,
    scaler_init: float,
    scaler_scale: float,
    alpha_init: float,
    alpha_scale: float,
    c_shift: float,
    gain: float,
) -> tuple[nn.Module, nn.Module]:
    embedder = HyperEmbedder(
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

    encoder = nn.Sequential(*lerp_blocks)

    return embedder, encoder


class SimbaV2Actor(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        in_features: int,
        hidden_features: int,
        action_dim: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        c_shift: float,
        gain: float = 1.0,
    ) -> None:
        super().__init__()

        self.embedder, self.encoder = create_simbav2_base(
            num_blocks=num_blocks,
            in_features=in_features,
            hidden_features=hidden_features,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            alpha_init=alpha_init,
            alpha_scale=alpha_scale,
            c_shift=c_shift,
            gain=gain,
        )

        self.predictor = HyperNormalTanhPolicy(
            in_features=hidden_features,
            hidden_features=hidden_features,
            action_dim=action_dim,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def forward(self, observations: torch.Tensor, temperature: float = 1.0):
        x = self.embedder(observations)
        x = self.encoder(x)
        mean, log_std = self.predictor(x, temperature)
        return mean, log_std


class SimbaV2Critic(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        in_features: int,  # NOTE: this must be obs_dim + action_dim
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

        self.embedder, self.encoder = create_simbav2_base(
            num_blocks=num_blocks,
            in_features=in_features,
            hidden_features=hidden_features,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            alpha_init=alpha_init,
            alpha_scale=alpha_scale,
            c_shift=c_shift,
            gain=gain,
        )

        self.predictor = HyperCategoricalValue(
            in_features=hidden_features,
            hidden_features=hidden_features,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    # NOTE: sb3 already concatenates obs and actions and passes it directly to this forward
    def forward(self, x: torch.Tensor):
        # x = torch.concatenate((observations, actions), dim=-1)  # original code uses dim = 1
        x = self.embedder(x)
        x = self.encoder(x)
        log_probs = self.predictor(x)
        return log_probs


class SimbaV2DiscreteActor(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        in_features: int,
        hidden_features: int,
        action_dim: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        c_shift: float,
        gain: float = 1.0,
    ) -> None:
        super().__init__()

        self.embedder, self.encoder = create_simbav2_base(
            num_blocks=num_blocks,
            in_features=in_features,
            hidden_features=hidden_features,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            alpha_init=alpha_init,
            alpha_scale=alpha_scale,
            c_shift=c_shift,
            gain=gain,
        )

        self.predictor = HyperCategorialPolicy(
            in_features=hidden_features,
            hidden_features=hidden_features,
            action_dim=action_dim,
            scaler_init=1.0,
            scaler_scale=1.0,
            gain=gain,
        )

    def forward(self, observations: torch.Tensor):
        x = self.embedder(observations)
        x = self.encoder(x)
        mean = self.predictor(x)
        return mean
