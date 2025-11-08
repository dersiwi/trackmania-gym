
from stable_baselines3.common.policies import BasePolicy
from neuronal_networks.extractors import make_tmn_extractor
from neuronal_networks.custom_extractor import AsyncActorCriticPolicy

import torch.nn as nn
import hydra
from omegaconf import OmegaConf, DictConfig, ListConfig
from gymnasium import spaces 

from functools import partial
from typing import Dict, Optional, Any
def get_policy(
    observation_space,
    policy_cfg,
    device: str,
    vision_model: partial[nn.Module],
    vision_model_kwargs: Optional[Dict[str, Any]] = None,
) -> tuple[str | BasePolicy, dict | None]:
    """
    Constructs a policy definition compatible with SB3 algorithms.

    Automatically selects the right TMN feature extractor (Box or Dict) 
    and forwards all kwargs to the extractor.

    Returns:
        tuple[str | BasePolicy, dict | None]
    """
    # Helper to build extractor kwargs from policy config
    def _build_extractor_kwargs(cfg):
        float_model = getattr(cfg, "float_net", None)
        activation_fn = hydra.utils.get_class(getattr(cfg, "activation_fn", {}).get("_target_", "torch.nn.ReLU"))
        last_activation_fn = hydra.utils.get_class(getattr(cfg, "last_activation_fn", {}).get("_target_", "torch.nn.Tanh"))
        out_dim = getattr(cfg, "extractors_out_dim", 64)
        normalized_image = getattr(cfg, "normalize_images", True)
        check_channels = getattr(cfg, "check_channels", True)

        return dict(
            vision_model=vision_model,
            vision_model_kwargs = vision_model_kwargs,
            out_dim=out_dim,
            device=device,
            float_model=float_model,
            activation_fn=activation_fn,
            last_activation_fn=last_activation_fn,
            normalized_image=normalized_image,
            check_channels=check_channels,
        )

    match policy_cfg.name:
        case "basic" | "dqn":
            # SB3 will pass observation_space automatically, do not include it here
            extractor_kwargs = _build_extractor_kwargs(policy_cfg)
            return policy_cfg.type, dict(
                features_extractor_class=make_tmn_extractor,
                features_extractor_kwargs=extractor_kwargs,
                normalize_images=getattr(policy_cfg, "normalize_images", True),
                share_features_extractor=getattr(policy_cfg, "share_features_extractor", True),
            )

        case "async_actor_critic":
            # For async AC, manually instantiate extractors with the proper subset of observation_space
            actor_obs = OmegaConf.to_object(ListConfig(policy_cfg.actor.observations))
            critic_obs = OmegaConf.to_object(ListConfig(policy_cfg.critic.observations))

            policy_features_extractor = make_tmn_extractor(
                observation_space=spaces.Dict({k: v for k, v in observation_space.items() if k in actor_obs}),
                **_build_extractor_kwargs(policy_cfg.actor)
            )

            value_features_extractor = make_tmn_extractor(
                observation_space=spaces.Dict({k: v for k, v in observation_space.items() if k in critic_obs}),
                **_build_extractor_kwargs(policy_cfg.critic)
            )

            # Optional MLP extractor
            if "mlp_extractor" in policy_cfg:
                net_arch = OmegaConf.to_object(DictConfig(policy_cfg.mlp_extractor.net_arch))
                act_fn = hydra.utils.get_class(policy_cfg.mlp_extractor.activation_fn._target_)
            else:
                net_arch = None
                act_fn = nn.Tanh  # fallback

            return AsyncActorCriticPolicy, dict(
                policy_features_extractor=policy_features_extractor,
                value_features_extractor=value_features_extractor,
                net_arch=net_arch,
                activation_fn=act_fn,
                normalize_images=getattr(policy_cfg, "normalize_images", True),
            )

        case _:
            raise ValueError(f"Policy type {policy_cfg.name} not known.")
