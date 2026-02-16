import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .simbav2_layers import HyperDense, Scaler
from .simbav2_networks import create_simbav2_base


# ripped from https://arxiv.org/pdf/2403.03950 Appendix A
class HLGaussLoss(nn.Module):
    def __init__(
        self,
        min_value: float,
        max_value: float,
        num_bins: int,
        sigma: float | None = None,
        sigma_to_bin_ratio: float | None = 2.0,  # from fig 6 of https://arxiv.org/html/2402.13425v2
        device: th.device | str = "auto",
    ):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.support = th.linspace(min_value, max_value, num_bins + 1, dtype=th.float32, device=device)
        mean_bin_size = (self.support[1:] - self.support[:-1]).mean().item()

        assert not (sigma is None and sigma_to_bin_ratio is None), "either `sigma` or `sigma_to_bin_ratio` is set but not both"
        self.sigma = sigma if sigma else sigma_to_bin_ratio * mean_bin_size
        assert self.sigma > 0.0

    def forward(self, logits: th.Tensor, target: th.Tensor) -> th.Tensor:
        return F.cross_entropy(logits, self.transform_to_probs(target))

    def transform_to_probs(self, target: th.Tensor) -> th.Tensor:
        cdf_evals = th.special.erf((self.support - target.unsqueeze(-1)) / (th.sqrt(th.tensor(2.0)) * self.sigma))
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return bin_probs / z.unsqueeze(-1)

    def transform_from_probs(self, probs: th.Tensor) -> th.Tensor:
        centers = (self.support[:-1] + self.support[1:]) / 2
        return th.sum(probs * centers, dim=-1)


class SimbaV2HLGaussCritic(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_bins: int,
        min_v: float,
        max_v: float,
        scaler_init: float,
        scaler_scale: float,
        gain: float = 1.0,
    ) -> None:
        super().__init__()
        self.w1 = HyperDense(in_features=in_features, out_features=hidden_features, gain=gain)
        self.scaler = Scaler(dim=hidden_features, init=scaler_init, scale=scaler_scale)
        self.w2 = HyperDense(in_features=hidden_features, out_features=num_bins, gain=gain)
        self.bias = nn.Parameter(data=th.zeros(size=(num_bins,)))
        self.register_buffer("bin_values", th.linspace(start=min_v, end=max_v, steps=num_bins + 1))

    def forward(self, x: th.Tensor):
        value = self.w1(x)
        value = self.scaler(value)
        logits = self.w2(value) + self.bias

        return logits


class HLGaussCritic(nn.Module):
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

        self.predictor = SimbaV2HLGaussCritic(
            in_features=hidden_features,
            hidden_features=hidden_features,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def forward(self, x: th.Tensor):
        x = self.embedder(x)
        x = self.encoder(x)
        logits = self.predictor(x)
        return logits
