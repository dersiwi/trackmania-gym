# this is a pytorch implementation of the nn architectur from the paper  "Hyperspherical Normalization for Scalable Deep Reinforcement Learning" https://arxiv.org/pdf/2502.15280

import torch
import torch.nn as nn


def l2normalize(x: torch.Tensor, axis: int, eps=1e-8) -> torch.Tensor:
    l2norm = torch.linalg.norm(x, ord=2, dim=axis, keepdim=True)
    x = x / torch.clamp(l2norm, min=eps)
    return x


# RSNorm implementation. section 3.2 input embedding
class RSNorm(nn.Module):
    def __init__(
        self, dim: int, eps: float = 1e-5, momentum: float | None = None
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        # Register buffers so they are saved in the state_dict but not trained by optimizer
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                batch_mean = x.mean(dim=0, keepdim=False)
                batch_var = x.var(dim=0, unbiased=False)
                batch_count = x.shape[0]

                if self.momentum is None:
                    # Method 1: Cumulative Moving Average (Exact historical mean/var)
                    total_count = self.count + batch_count
                    ratio = batch_count / total_count

                    delta = batch_mean - self.mean
                    new_mean = self.mean + delta * ratio

                    m_a = self.var * self.count
                    m_b = batch_var * batch_count
                    M2 = m_a + m_b + delta**2 * self.count * ratio
                    new_var = M2 / total_count

                    self.mean.copy_(new_mean)
                    self.var.copy_(new_var)
                    self.count.copy_(total_count)

                else:
                    # Method 2: Exponential Moving Average
                    self.mean.lerp_(batch_mean, self.momentum)
                    self.var.lerp_(batch_var, self.momentum)

        return (x - self.mean) / torch.sqrt(self.var + self.eps)

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.dim}, eps={self.eps}, momentum={self.momentum})"


# section 4.4 Scaler
class Scaler(nn.Module):
    def __init__(self, dim: int, init: float = 1.0, scale: float = 1.0) -> None:
        super().__init__()
        self.init = init
        self.scale = scale
        self.scaler = nn.Parameter(
            data=torch.ones(dim) * self.scale, requires_grad=True
        )
        assert self.scale != 0.0, "You can't initialize scale to 0"
        forward_scaler = self.init / self.scale
        self.register_buffer("forward_scaler", torch.tensor(forward_scaler))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s: torch.Tensor = self.scaler * self.forward_scaler
        return s * x


# pytorch linear interpolation module with learnable interpolation coefficient
class LERP(nn.Module):
    def __init__(self, dim: int, init: float = 1.0, scale: float = 1.0) -> None:
        super().__init__()
        self.alpha = Scaler(dim=dim, init=init, scale=scale)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # (1-a) * x1 + a * x2 = x1 - a*x1 + a*x2 = x1 + a*(x2-x1)
        lerp = x1 + self.alpha(x2 - x1)
        normed = l2normalize(x=lerp, axis=-1)
        return normed


# section 4.4 Weight: Just a linear layer with ortho init and bias set to false
class HyperDense(nn.Module):
    def __init__(self, in_features: int, out_features: int, gain: float = 1.0) -> None:
        super().__init__()
        self.in_dim = in_features
        self.out_dim = out_features
        self.gain = gain
        # NOTE: bias must be set to false see https://github.com/DAVIAN-Robotics/SimbaV2/blob/master/scale_rl/agents/simbaV2/simbaV2_layer.py#L31
        self.w = nn.Linear(
            in_features=in_features, out_features=out_features, bias=False
        )
        nn.init.orthogonal(self.w, gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x)


# the emdding block from figure 3 + 4.1
# NOTE: we include the RSNorm in this implementation but the original implementation uses gym wrappers see https://github.com/DAVIAN-Robotics/SimbaV2/tree/86899c277cdc697b2b02d827243de1ea93f20a1d/scale_rl/agents/wrappers
class HyperEmbedder(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        scaler_init: float,
        scaler_scale: float,
        c_shift: float,
        gain: float = 1.0,
    ) -> None:
        super().__init__()
        self.w = HyperDense(
            in_features=in_features, out_features=out_features, gain=gain
        )
        self.scaler = Scaler(dim=out_features, init=scaler_init, scale=scaler_scale)
        self.register_buffer("c_shift", torch.tensor(c_shift, dtype=torch.float))

    def forward(self, x: torch.Tensor):
        new_axis = torch.ones((x.shape[:-1] + (1,))) * self.c_shift
        x = torch.concatenate([x, new_axis], dim=-1)
        x = l2normalize(x, axis=-1)
        x = self.w(x)
        x = self.scaler(x)
        x = l2normalize(x, axis=-1)
        return x


# the mlp used in the LERP residual block before the LERP happens. see Figure 3
class HyperMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        scaler_init: float,
        scaler_scale: float,
        gain: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.w1 = HyperDense(
            in_features=in_features, out_features=hidden_features, gain=gain
        )
        self.scaler = Scaler(dim=hidden_features, init=scaler_init, scale=scaler_scale)
        self.w2 = HyperDense(
            in_features=hidden_features, out_features=out_features, gain=gain
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = self.scaler(x)
        # `eps` is required to prevent zero vector.
        x = nn.functional.relu(x) + self.eps
        x = self.w2(x)
        x = l2normalize(x, axis=-1)
        return x


# the residual block from Figure 3
class HyperLERPBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        gain: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.mlp = HyperMLP(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            eps=eps,
        )
        self.lerp = LERP(dim=out_features, init=alpha_init, scale=alpha_scale)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.mlp(x)
        x = self.lerp(residual, x)  # (1-res)*a + a*x Eq:12
        x = l2normalize(x, axis=-1, eps=self.eps)
        return x


class SimbaV2ResidualBlock(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
