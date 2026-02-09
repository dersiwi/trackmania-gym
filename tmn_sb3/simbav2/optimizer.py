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


if __name__ == "__main__":
    import sys
    import os

    # add the repo root to Python path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

    from tmn_sb3.simbav2.simbav2_layers import HyperDense
    import torch.nn as nn
    for _ in range(5):
        in_features, out_features = 10, 5
        hyper_layer = HyperDense(in_features, out_features)
        normal_layer = nn.Linear(in_features, out_features, bias=False)

        optimizer = UnitAdam(list(hyper_layer.parameters()) + list(normal_layer.parameters()), lr=0.1)

        dummy_input = torch.randn(1, in_features)
        loss = hyper_layer(dummy_input).sum() + normal_layer(dummy_input).sum()
        loss.backward()

        print("--- Norms Before Step ---")
        initial_hyper_norms = torch.linalg.norm(hyper_layer.w.weight, ord=2, dim=1)
        print(f"HyperDense row norms:\n{initial_hyper_norms}")

        # this should trigger the projection
        optimizer.step()

        print("\n--- Norms After Step ---")
        final_hyper_norms = torch.linalg.norm(hyper_layer.w.weight, ord=2, dim=1)
        final_normal_norms = torch.linalg.norm(normal_layer.weight, ord=2, dim=1)

        print(f"HyperDense row norms (should be 1.0):\n{final_hyper_norms}")
        print(f"Normal layer row norms (should NOT be 1.0):\n{final_normal_norms}")

        is_success = torch.allclose(final_hyper_norms, torch.ones_like(final_hyper_norms), atol=1e-6)
        print(f"\nUnit-Norm Projection Successful: {is_success}")
