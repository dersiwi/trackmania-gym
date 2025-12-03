
from functools import partial

import torch.nn as nn
import hydra

from omegaconf import DictConfig, OmegaConf,ListConfig
from itertools import chain



from sb3_contrib.qrdqn.qrdqn import QRDQN

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm


from configs.config import TrainConfig
from typing import Any, Dict

from dataclasses import replace as data_cls_replace
from tmn_sb3.policies.async_policy import build_async_actor_critic_policy
from utils.hydra_wandb_utils import secure_attribute_retrieval

from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
from tmn_sb3.policies.async_policy import AsyncActorCriticPolicy
from neural_networks.lr_schedulers import LR_Scheduler
from neural_networks.extractors.extractors import ExtractorConfig
from neural_networks.extractors import make_tmn_extractor


def print_model_params(model : BaseAlgorithm):
    """"Prints parametrs of the given model"""
    print("\nModel Parameters\n" + "-"*40)
    for name, param in model.policy.named_parameters():
        print(f"{name:<75} {str(tuple(param.shape)):<25}  grad={param.requires_grad}")

def get_model_from_config(
        cfg : TrainConfig,
        tm_env : TMNF_Single_Agent_Env,
        print_params : bool = False,
        run_id:str = "test",
        load_model_path: str | None = None,
        load_replay_buffer_path: str | None = None
        ) -> BaseAlgorithm:

    """
    This function builds a policy and creates the corresponding SB3 model (e.g., PPO, DQN, A2C, etc.) 
    dynamically based on a configuration object. It supports asynchronous actor-critic models. Also:

    - Supports both actor-critic and value-based methods.
    - The model can be configured to share or separate feature extractors between actor and critic networks.
    - Uses Hydra for instantiating models, schedulers, and other configurable components.

    Args:
        cfg (TrainConfig) : Global training configuration containing all parameters related to models,  environments, algorithms, and policies.
        tm_env (TMNF_Single_Agent_Env) :             The Gymn environment used for training the agent.
        print_params  (bool) : If True, prints the shapes of the neural network weights for inspection. Defaults to False.
        run_id (str) : Identifier for the current run. Used for TensorBoard logging. Defaults to "test".
        load_model_path (str) : Path to a saved model. If provided, the model parameters  are loaded into the instantiated model.
        load_replay_buffer_path (str) : Path to a previously saved replay buffer. If provided and the algorithm  is off-policy (e.g., DQN, SAC), the replay buffer is loaded as well.

    Returns
    -------
    model : BaseAlgorithm
        The fully constructed and configured Stable-Baselines3 algorithm ready for training or evaluation.
    """
    device = cfg.platforms.device
    normalized_images = cfg.rl_env.env.normalize_images

    # Prepare Vision Model
    vision_model = (hydra.utils.instantiate(cfg.models) if secure_attribute_retrieval(lambda: cfg.rl_env.env.obs_have_imgs, True) else None) 

    vision_model_kwargs = (
        OmegaConf.to_container(cfg.models.args, resolve=True)
        if secure_attribute_retrieval(lambda: cfg.models.args, False)
        else {}
    )

    # extract algo params
    algorithm_params = OmegaConf.to_container(cfg.sb3.algorithm_params, resolve=True)

    #Create policy related things like extractors 
    policy_cfg = cfg.policy
    policy_type = policy_cfg.type 
    policy_kwargs = None

    base_ext_config = ExtractorConfig.create(policy_cfg, cfg, vision_model, vision_model_kwargs, device)

    if policy_cfg.name in {"basic","dqn"}:

        feature_extrac_kwargs = base_ext_config
        policy_kwargs = dict(
            features_extractor_class=make_tmn_extractor,
            features_extractor_kwargs=feature_extrac_kwargs.to_dict(),
            normalize_images= normalized_images,
        )
        # dqn style algos are not type of actor critic methods so they dont have that field
        if policy_cfg.name != "dqn": policy_kwargs.update(dict(share_features_extractor=policy_cfg.share_features_extractor))

    elif policy_cfg.name == "async_actor_critic": 

        actor_config = data_cls_replace(
            base_ext_config,
            float_model=secure_attribute_retrieval(lambda: policy_cfg.actor.float_net, None),
            activation_fn=hydra.utils.get_class(policy_cfg.actor.activation_fn._target_),
            last_activation_fn=hydra.utils.get_class(policy_cfg.actor.last_activation_fn._target_),
            out_dim = policy_cfg.actor.extractors_out_dim,
        )
        
        value_config = data_cls_replace(
            base_ext_config,
            float_model=secure_attribute_retrieval(lambda: policy_cfg.critic.float_net, None),
            activation_fn=hydra.utils.get_class(policy_cfg.critic.activation_fn._target_),
            last_activation_fn=hydra.utils.get_class(policy_cfg.critic.last_activation_fn._target_),
            out_dim =  policy_cfg.critic.extractors_out_dim,
        )
        
        policy_type, policy_kwargs = build_async_actor_critic_policy(
                observation_space= tm_env.observation_space,
                actor_observations=  OmegaConf.to_object(ListConfig(policy_cfg.actor.observations)),
                actor_extractor_kwargs= actor_config.to_dict(),
                critic_observations =  OmegaConf.to_object(ListConfig(policy_cfg.critic.observations)),
                critic_extractor_kwargs= value_config.to_dict(),
                net_arch= OmegaConf.to_object(DictConfig(policy_cfg.mlp_extractor.net_arch)) if secure_attribute_retrieval(lambda: policy_cfg.mlp_extractor.net_arch,False) else None,
                activation_fn = hydra.utils.get_class(policy_cfg.mlp_extractor.activation_fn._target_) if secure_attribute_retrieval(lambda: policy_cfg.mlp_extractor.activation_fn._target_,False) else nn.Tanh,
                normalize_images = normalized_images,
                )

    else: raise ValueError(f"Unknown policy name: {policy_cfg.name!r}") 

    model_args = dict(
    policy = policy_type,
    env = tm_env,
    tensorboard_log = run_id,
    device = device,
    **algorithm_params
    )
     # Only include policy_kwargs if they exist
    if policy_kwargs: model_args["policy_kwargs"] = policy_kwargs

    lr : LR_Scheduler = hydra.utils.instantiate(cfg.lr_scheduler)
    model_args["learning_rate"] = lr.get_scheduler()

    model_constructor = hydra.utils.instantiate(cfg.sb3.constructor)
    model : BaseAlgorithm = model_constructor(**model_args)
    
    if print_params:
        print_model_params(model)

    if load_model_path: 
        # set_parameters() operates in in-place. If we would use model.load() we would 
        # need to reassign it to the model variable since the method doesn't work in-place
        model.set_parameters(load_model_path)
        print(f"Loading model from {load_model_path}...")

        if isinstance(model,OffPolicyAlgorithm) and load_replay_buffer_path:
            model.load_replay_buffer(load_replay_buffer_path)
            print(f"Loading replay buffer from {load_replay_buffer_path}...")
    
    return model
