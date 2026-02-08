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
        # update gradients with normal adam 
        loss = super().step(closure)
        
        # now project the weights onto the unit hemisphere
        with torch.no_grad():
            for group in self.param_groups:
                for p in group["params"]:
                    if p is None:
                        continue

                    if not getattr(p, "_hyper_dense", False):
                        continue

                    if p.ndim == 2:
                        axis = 0
                    elif p.ndim == 3:
                        axis = 1
                    else:
                        continue 

                    norm = torch.linalg.norm(p, ord=2, dim=axis, keepdim=True).clamp(min=self.eps)

                    p.div_(norm)

        return loss
