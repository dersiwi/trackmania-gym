
from stable_baselines3.common.base_class import BaseAlgorithm

def print_model_params(model : BaseAlgorithm):
    for name, param in model.policy.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")