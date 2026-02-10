import sys
import os

import torch 
# add the repo root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from neural_networks.vision_encoder.simbav2_cnn import UnitConv2D, SimbaV2VisionModel
from tmn_sb3.simbav2.optimizer import UnitAdam

def test_unit_conv2d_normalization():
    print("Testing UnitConv2D...")
    img_shape = (3, 64, 64)
    out_dim = 10 
    model = SimbaV2VisionModel(out_dim=out_dim, img_shape=img_shape)
    
    optimizer = UnitAdam(model.parameters(), lr=0.1)

    x = torch.randn(2, *img_shape)
    
    output = model(x)
    loss = output.sum()
    loss.backward()
    optimizer.step()

    found_hyper_params = False
    for param_name, param in model.named_parameters():
        if getattr(param, "_hyper_dense", False):
            found_hyper_params = True
            
            # For Conv2D, we normalize per output filter.
            # We flatten all dimensions except the first (out_channels)
            flattened_weights = param.view(param.shape[0], -1)
            norms = torch.linalg.norm(flattened_weights, ord=2, dim=1)
            
            # Check if all filters have unit norm
            is_unit = torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
            status = "PASSED" if is_unit else "FAILED"
            
            print(f"  -> Parameter '{param_name}' (Shape {list(param.shape)}): {status}")
            print(f"  -> Mean Norm: {norms.mean().item():.4f}")

            if not is_unit:
                raise ValueError(f"Unit norm check failed for {param_name}")

    if not found_hyper_params:
        print("  -> WARNING: No hyper-dense parameters found!")
    
    print("-" * 30)

if __name__ == "__main__":
    try:
        test_unit_conv2d_normalization()
        print("CONV2D NORMALIZATION TEST PASSED.")
    except Exception as e:
        print(f"TEST FAILED: {e}")
