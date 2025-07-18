from stable_baselines3.common.policies import ActorCriticPolicy,BasePolicy
from neuronal_networks.lr_schedulers import LR_Scheduler
from neuronal_networks.custom_extractor import TMN_Extractor, AsyncActorCriticPolicy

import torch.nn as nn
import hydra
from omegaconf import OmegaConf,DictConfig,ListConfig
from gymnasium.spaces import Dict

def get_policy(
        observation_space,
        policy_cfg,
        device:int,
        vision_model:nn.Module) -> tuple[str | BasePolicy, dict | None]:
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
    #policy_obj = OmegaConf.to_object(DictConfig(policy_cfg))

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
            actor_obs = OmegaConf.to_object(ListConfig(policy_cfg.actor.observations))
            critic_obs = OmegaConf.to_object(ListConfig(policy_cfg.critic.observations))

            policy_features_extractor = TMN_Extractor(
                observation_space = Dict({k: v for k, v in observation_space.items() if k in actor_obs}),
                vision_model = vision_model,
                out_dim =  policy_cfg.actor.extractors_out_dim,
                device= device,
                float_model= policy_cfg.actor.float_net,
                activation_fn =  hydra.utils.get_class(policy_cfg.actor.activation_fn._target_),
                last_activation_fn =  hydra.utils.get_class(policy_cfg.actor.last_activation_fn._target_)
            )

            value_features_extractor = TMN_Extractor(
                observation_space = Dict({k: v for k, v in observation_space.items() if k in critic_obs}),
                vision_model = vision_model,
                out_dim =  policy_cfg.critic.extractors_out_dim,
                device= device,
                float_model= policy_cfg.critic.float_net,
                activation_fn =  hydra.utils.get_class(policy_cfg.critic.activation_fn._target_),
                last_activation_fn =  hydra.utils.get_class(policy_cfg.critic.last_activation_fn._target_)
            )
            if "mlp_extractor" in policy_cfg:
                net_arch = OmegaConf.to_object(DictConfig(policy_cfg.mlp_extractor.net_arch))
                act_fn = hydra.utils.get_class(policy_cfg.mlp_extractor.activation_fn._target_)
            else:
                net_arch = None
                act_fn = nn.Tanh  #fallback this does not really matter since it will never be used 
            return AsyncActorCriticPolicy, dict(
                policy_features_extractor= policy_features_extractor,
                value_features_extractor= value_features_extractor,
                net_arch=net_arch,
                activation_fn=act_fn
            )
        case _ :
            raise ValueError(f"Policy type {policy_cfg.name} not known.") 
