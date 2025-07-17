from stable_baselines3.common.policies import ActorCriticPolicy,BasePolicy
from neuronal_networks.lr_schedulers import LR_Scheduler
from neuronal_networks.custom_extractor import TMN_Extractor
import torch.nn as nn


def get_policy(policy_cfg, device:int, vision_model:nn.Module) -> tuple[str | BasePolicy, dict | None]:
    """
    Constructs a policy definition compatible with SB3 algorithms.

    Depending on the `policy_cfg.name`, this function either:
    - Returns a fully constructed custom policy object, or
    - Returns a tuple of (policy_type, policy_kwargs) suitable for use in SB3 model constructors.

    Supported policy types:
    ------------------------
    - "basic":
        A simple, general-purpose policy using the `TMN_Extractor` as the feature extractor.
        Suitable for both on-policy and off-policy algorithms.

    - "async_ACOT_CRITIC":
        support of asynchronous actor-critic architecture.

    Parameters:
    -----------
    policy_cfg : Any
        Configuration object specifying the policy type and its parameters.

    device : int
        The target device (e.g., GPU ID) for model placement.

    vision_model : nn.Module
        A vision backbone (e.g., CNN or transformer) to be used inside the feature extractor.

    Returns:
    --------
    tuple[str | BasePolicy, dict | None]
        - If using a built-in or named SB3 policy: returns (policy_name, policy_kwargs).
        - If using a fully constructed policy class: returns (policy_instance, None).
    """
    
    match policy_cfg.name:
        case "basic":
            return policy_cfg.type ,dict(
                features_extractor_class=TMN_Extractor,
                share_features_extractor = policy_cfg.share_features_extractor,
                features_extractor_kwargs= dict( 
                    vision_model = vision_model,
                    device= device,
                    out_dim =  policy_cfg.extractors_out_dim)
                )
        
        case "async_actor_critic":
            0
        case _ :
            raise ValueError(f"Policy type {policy_cfg.name} not known.") 
