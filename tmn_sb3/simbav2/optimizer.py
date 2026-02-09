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

    from tmn_sb3.simbav2.simbav2_layers import (
        HyperDense,
        HyperMLP,
        HyperCategoricalValue,
        HyperNormalTanhPolicy,
        HyperEmbedder,
        HyperLERPBlock,
    )

    def run_normalization_test(name, model, in_dim):
        print(f"Testing {name}...")
        optimizer = UnitAdam(model.parameters(), lr=0.1)

        # Create dummy input and perform a training step
        x = torch.randn(2, in_dim)
        # Use a dummy loss that forces a gradient on everything
        output = model(x)
        if isinstance(output, tuple):  # Handle Actor returning (mean, std)
            loss = sum(o.sum() for o in output)
        else:
            loss = output.sum()

        loss.backward()
        optimizer.step()

        # Iterate through all parameters and check those with the flag
        found_hyper_params = False
        for param_name, param in model.named_parameters():
            if getattr(param, "_hyper_dense", False):
                found_hyper_params = True
                # Compute row-wise L2 norms
                norms = torch.linalg.norm(param, ord=2, dim=1)
                # Check if all norms are 1.0
                is_unit = torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

                status = "PASSED" if is_unit else "FAILED"
                print(f"  -> Parameter '{param_name}': {status} (Norm: {norms[0].item():.4f}...)")

                if not is_unit:
                    raise ValueError(f"Unit norm check failed for {param_name} in {name}")

        if not found_hyper_params:
            print(f"  -> WARNING: No hyper-dense parameters found in {name}")
        print("-" * 30)

    IN, OUT, HIDDEN = 16, 8, 32

    # Test HyperDense
    run_normalization_test("HyperDense", HyperDense(IN, OUT), IN)
    # Test HyperMLP
    run_normalization_test("HyperMLP", HyperMLP(IN, OUT, HIDDEN, 0.1, 0.1), IN)
    # Test HyperEmbedder
    run_normalization_test("HyperEmbedder", HyperEmbedder(IN, OUT, 0.1, 0.1, 3.0), IN)
    # Test HyperLERPBlock
    # Note: in_features must equal out_features for residual addition
    run_normalization_test("HyperLERPBlock", HyperLERPBlock(OUT, OUT, HIDDEN, 0.1, 0.1, 0.5, 0.1), OUT)

    # 5. Test HyperNormalTanhPolicy (Actor)
    run_normalization_test("HyperNormalTanhPolicy", HyperNormalTanhPolicy(IN, HIDDEN, 4, 0.1, 0.1), IN)

    # 6. Test HyperCategoricalValue (Critic)
    run_normalization_test("HyperCategoricalValue", HyperCategoricalValue(IN, HIDDEN, 101, -5, 5, 0.1, 0.1), IN)

    print("\nALL SIMBA-V2 COMPONENTS PASSED HYPERSPHERICAL WEIGHT NORMALIZATION TESTS.")
