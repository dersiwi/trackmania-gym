from torch.optim import Adam

import torch
from torch import Tensor

from torch.optim.optimizer import (
    Optimizer,
    ParamsT,
)


class UnitAdam(Adam):
    """
    This is an implementation of the Adam optimizer that automatically projects the weights onto the unit-norm hypersphere
    after each gradient update. This method is used in SimBaV2 (https://arxiv.org/pdf/2502.15280) (Section 1, Hyperspherical
    Weight Normalization)
    """

    eps = 1e-8

    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        *,
        foreach: bool | None = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
    ):
        super().__init__(
            params,
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )

    def step(self, closure=None):
        # Standard Adam Update. After this the gradients have been updated
        loss = super().step(closure)

        # Projection Step
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    if not getattr(p, "_hyper_dense", False):
                        continue

                    if p.ndim == 2:
                        dim = 1
                    elif p.ndim == 3:
                        dim = (1, 2)
                    elif p.ndim == 4:
                        dim = (1, 2, 3)
                    else:
                        continue

                    norm = torch.linalg.norm(p, ord=2, dim=dim, keepdim=True)
                    p.div_(torch.maximum(norm, torch.tensor(1e-8, device=p.device)))

        return loss
