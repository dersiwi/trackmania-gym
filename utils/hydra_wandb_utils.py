from functools import partial

import torch.nn as nn
import os
import hydra
import wandb
from wandb.wandb_run import Run

from omegaconf import DictConfig, OmegaConf, open_dict, ListConfig
from omegaconf.errors import ConfigAttributeError
from itertools import chain

from tmn_sb3.policies.async_policy import AsyncActorCriticPolicy
from neuronal_networks.lr_schedulers import LR_Scheduler

from sb3_contrib.qrdqn.qrdqn import QRDQN

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy,BasePolicy
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from trackmania_env.utils.return_tracker import ReturnLogCallback
from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env

from configs.config import TrainConfig
from typing import Callable, Any, Dict

from neuronal_networks.extractors.extractors import ExtractorConfig

from neuronal_networks.extractors import make_tmn_extractor

from dataclasses import replace as data_cls_replace

from tmn_sb3.policies.async_policy import build_async_actor_critic_policy

def secure_attribute_retrieval(getter : Callable, default : any = None):
    """Securely returns an attribute from the config.
    Args:
        getter (Callable)   : Lambda function retrieving attribute from config : `lambda: cfg.gmi` (e.g.)
        default (any)       : Default value to be returned if getter fails
    """
    try:    
        return getter()
    except ConfigAttributeError:
        return default

def print_model_params(model : BaseAlgorithm):
    """"Prints parametrs of the given model"""
    print("\nExtractor, Policy and Critic architecturs:\n" + "-"*30)

    for name, param in model.policy.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")

    print("\nFeature Extractor Parameters:\n" + "-"*30)
    if isinstance(model.policy,AsyncActorCriticPolicy):
        for name, param in chain(
            model.policy.policy_features_extractor.named_parameters(),
model.policy.value_features_extractor.named_parameters(),
            model.policy.mlp_extractor.named_parameters()):
            print(name, param.shape)
    elif isinstance(model,QRDQN):
        for name, param in model.quantile_net.features_extractor.named_parameters():
            print(f"{name}: requires_grad = {param.requires_grad}")
    else:
        for name, param in chain(model.policy.features_extractor.named_parameters(),model.policy.mlp_extractor.named_parameters()):
            print(f"{name}: requires_grad = {param.requires_grad}")

    
    if isinstance(model.policy, ActorCriticPolicy):
        print("\nActor- and Value-Networks Parameters:\n" + "-"*30)
        for name, param in chain(model.policy.action_net.named_parameters(),model.policy.value_net.named_parameters()):
            print(f"{name}: requires_grad = {param.requires_grad}")
        print("\n[INFO] Checking whether the actor and critic are using the same feature extractor:\n" + "-"*30)
        if isinstance(model.policy,AsyncActorCriticPolicy):
            actor_id = id(model.policy.policy_features_extractor)
            critic_id = id(model.policy.value_features_extractor)
        else:   
            actor_id = id(model.policy.pi_features_extractor)
            critic_id = id(model.policy.vf_features_extractor)

        print("Actor Feature Extractor ID:", actor_id)
        print("Critic Feature Extractor ID:", critic_id)

        if actor_id != critic_id:
            print("Actor and Critic are using DIFFERENT feature extractors.")
        else:
            print("Actor and Critic are sharing the SAME feature extractor.")



def get_vision_model(cfg : TrainConfig, img_shape, extractor_out_dim : int) -> nn.Module:
    """Create and return vision model according to configuration"""
    vision_model_constructor = hydra.utils.instantiate(cfg.models)
    in_color_channels = img_shape[0]
    #this may not be pretty, but not having the correct input channels has caused headaches
    expected_inchannel = 1 if cfg.rl_env.obs_manager.colorspace == "grayscale" else 3
    assert in_color_channels == expected_inchannel, f"Expected {expected_inchannel} color channels, got {in_color_channels}"

    vision_model : nn.Module = vision_model_constructor(img_shape = img_shape, out_dim = extractor_out_dim)
    
    return vision_model

def _create_base_extractor_config(
    policy_cfg: DictConfig,
    cfg: DictConfig,
    tm_env: TMNF_Single_Agent_Env,
    vision_model: partial[nn.Module],
    vision_model_kwargs: Dict[str,Any],
    device: str
) -> ExtractorConfig:
    """Creates a base ExtractorConfig with all shared/common parameters."""
    
    activation_fn_class = secure_attribute_retrieval(
        lambda: hydra.utils.get_class(policy_cfg.activation_fn._target_), nn.ReLU
    )
    last_activation_fn_class = secure_attribute_retrieval(
        lambda: hydra.utils.get_class(policy_cfg.last_activation_fn._target_), nn.Identity
    )
    
    return ExtractorConfig(
        vision_model=vision_model,
        vision_model_kwargs=vision_model_kwargs, 
        normalized_image=cfg.rl_env.env.normalize_images,
        out_dim=secure_attribute_retrieval(lambda: policy_cfg.extractors_out_dim,64),
        check_channels=cfg.rl_env.env.check_channels,
        
        float_model=secure_attribute_retrieval(lambda: policy_cfg.float_net, None), 
        activation_fn=activation_fn_class,
        last_activation_fn=last_activation_fn_class,
        device = device 
    )

def get_models(
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

    Parameters
    ----------
    cfg : TrainConfig
        Global training configuration containing all parameters related to models, 
        environments, algorithms, and policies.
    tm_env : TMNF_Single_Agent_Env
        The Gymn environment used for training the agent.
    print_params : bool, optional
        If True, prints the shapes of the neural network weights for inspection.
        Defaults to False.
    run_id : str, optional
        Identifier for the current run. Used for TensorBoard logging.
        Defaults to "test".
    load_model_path : str or None, optional
        Path to a saved model. If provided, the model parameters 
        are loaded into the instantiated model.
    load_replay_buffer_path : str or None, optional
        Path to a previously saved replay buffer. If provided and the algorithm 
        is off-policy (e.g., DQN, SAC), the replay buffer is loaded as well.

    Returns
    -------
    model : BaseAlgorithm
        The fully constructed and configured Stable-Baselines3 algorithm ready for training or evaluation.
    """
    device = cfg.platforms.device
    normalized_images = cfg.rl_env.env.normalize_images

    #Prepare Vision Model
    vision_model = (
        hydra.utils.instantiate(cfg.models) if secure_attribute_retrieval(lambda: cfg.rl_env.env.obs_have_imgs, True) else None) 

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

    base_ext_config = _create_base_extractor_config(policy_cfg, cfg, tm_env, vision_model, vision_model_kwargs, device)

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
            out_dim =  policy_cfg.actor.extractors_out_dim,
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


def init_and_login_wandb(cfg : TrainConfig, wandbdir : str = "wandb",run_id = None, resume = None ) -> tuple[Run | None, str]:
    """Instanciates and logs into weights and biases (wandb), if specified in configuration (cfg.wandb.use).
    After login, returns tuple of Run-instance and run-id 
    
    If cfg.wandb.use is False, the returned Run is None.

    :param cfg: The configuration object containing wandb settings.
    :param wandbdir: The directory where wandb run data will be stored.
    :param run_id: The ID of a previous run to resume.
    :param resume: The resume behavior for the wandb run (e.g., "allow", "must", "never")
    """
    #run_id = run_id or ""
    if cfg.wandb.use:
        wandb.login()
        wandb_conf = OmegaConf.to_container(cfg, resolve=True,throw_on_missing=True)
        run = wandb.init(
            entity=cfg.wandb.entity, 
            project=cfg.wandb.project,
            sync_tensorboard=True, 
            monitor_gym=True,  
            save_code=True,
            dir = wandbdir,
            config=wandb_conf,
            id = run_id,
            resume= resume)
        run_id = run.id
        wandb.config.update(wandb_conf,allow_val_change=True) # if run gets resumed then this has to updated and not overriden via wandb.config = wandbn_conf
        return run, run_id
    else:
        return None, ""
    

def save_model(model : BaseAlgorithm, run_dir : str) -> None:
    """Saves given model in given run-directory. Also saves replay buffer if the model has one."""
    steps_trained = model.num_timesteps

    savepoint_dir = os.path.join(run_dir, "savepoint")
    os.makedirs(savepoint_dir, exist_ok=True)

    model_save_path = model_save_path or os.path.join(savepoint_dir, "model.zip")
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

        # Save the replay buffer if it exists
    if hasattr(model, 'save_replay_buffer'):
        replay_buffer_save_path = replay_buffer_save_path or os.path.join(savepoint_dir, "replay_buffer.pkl")
        model.save_replay_buffer(replay_buffer_save_path)
        print(f"Replay buffer saved to {replay_buffer_save_path}")

    return steps_trained, model_save_path, replay_buffer_save_path


from hydra.utils import to_absolute_path

def load_and_merge_yaml(cfg : TrainConfig, yaml_to_merge : str) -> TrainConfig:
    """Loads a yaml and merges it with cfg
    Args:
        cfg (TrainConfig)   : Configuration provided by hydra
        yaml_to_merge (str) : Path to external yaml file to merge with yaml
    Returns:
        Merged (TrainConcig): Merged hydra-config with external yaml."""
    yaml = OmegaConf.load(to_absolute_path(yaml_to_merge))
    with open_dict(cfg):
        merged = OmegaConf.merge(cfg, yaml)
    return merged

def load_and_merge_platform(cfg : TrainConfig) -> TrainConfig:
    """Load and merge the standalone platform-yaml and merge it into the global 'cfg'-yaml."""
    platform_path = None
    platform_default_path = "configs/platforms.yaml"
    try:
        platform_path = cfg.platforms_config_path
    except ConfigAttributeError as e:
        print(f"[!!] Platform-path attribute missing in cfg; config too old? Trying default path : {platform_default_path}")
        platform_path = platform_default_path
    assert not platform_path is None
    return load_and_merge_yaml(cfg, platform_path)


from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from trackmania_env.rewards.reward_calculation import RewardLogCallback, AccumRewardLogCallback
import glob


class BeforeAndAfterTraining:
    """
    General
    --------
    This is a utility class, which contains methods and code that is executed before and after a training.
    Its main purpose is to perform hydra initailizations and wandb login etc. Usage : 
        1) baaf.before_training()
        2) do the training 
        3) baaf.after_training()

    Utilities
    ---------
    This class also provides variables and utilities used by hydra or wanbd
        1) get_callbacks_for_training()
        2) get_tensorboard_login_identifier()
    """

    def __init__(self, hydra_run_dir : str, cfg : TrainConfig, run_id = None, resume = None ):
        self.hydra_run_dir = hydra_run_dir
        self.cfg = cfg
        self.run_id = run_id
        self.resume = resume


    def before_training(self) -> None:
        """Create necessary paths and initate wandb-login."""
        model_dir = os.path.join(self.hydra_run_dir, "models")
        self.best_model_path = os.path.join(model_dir, "best_model")
        self.checkpoint_path = os.path.join(model_dir, "checkpoints")
        os.makedirs(self.best_model_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # Start Weights and Biases login
        self.run, self.run_id = init_and_login_wandb(self.cfg, wandbdir=self.hydra_run_dir,run_id= self.run_id,resume= self.resume)
        self.run_id_in_hydra_log_dir = os.path.join(self.hydra_run_dir, self.run_id)


    def get_tensorboard_login_identifier(self) -> str:
        """identifier for the run which gets used for tensorboard login"""
        return self.run_id_in_hydra_log_dir


    def get_callbacks_for_training(self, tm_env : TMNF_Single_Agent_Env) -> CallbackList:
        """Create wandb-callbacks for the RL-model to log custom metrics.
        Args : tm_env (gym.Env)"""
        # Eval Callback – save best model based on reward
        eval_callback = EvalCallback(
            tm_env,
            best_model_save_path=self.best_model_path,
            log_path=os.path.join(self.hydra_run_dir, "eval_logs"),
            eval_freq=self.cfg.wandb.eval_freq,
            deterministic=True,
            render=False,
        )

        # Checkpoint Callback – save model every N steps
        checkpoint_callback = CheckpointCallback(
            save_freq=self.cfg.wandb.checkpoint_freq,
            save_path=self.checkpoint_path,
            name_prefix="checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        callbacklist = [eval_callback, checkpoint_callback]
        #if cfg.wandb.use:
        #    callbacklist.append(hydra.utils.instantiate(cfg.wandb_callbacks)(model_save_path=RUN_ID_IN_HYDRA_LOG_DIR))
        #    callbacklist.append(RewardLogCallback())
            
        callback : CallbackList= CallbackList(callbacklist)
        if self.cfg.wandb.use:
            callback.callbacks.extend([hydra.utils.instantiate(self.cfg.wandb_callbacks)(model_save_path=self.run_id_in_hydra_log_dir), AccumRewardLogCallback(), ReturnLogCallback()])

        return callback
    

    def after_training(self, model : BaseAlgorithm):
        """Finish wandb run, save and upload models"""
        final_model =  os.path.join(self.hydra_run_dir, "model.zip")
        best_model = os.path.join(self.best_model_path, "best_model.zip")
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_path, "*.zip"))
        hydra_dir = os.path.join(self.hydra_run_dir, ".hydra")
        # always save final model
        model.save(os.path.join(self.hydra_run_dir, "model"))

        if self.cfg.wandb.use:
            # Upload best model
            if os.path.exists(best_model):
                best_artifact = wandb.Artifact("best_model", type="model")
                best_artifact.add_file(best_model)
                self.run.log_artifact(best_artifact)

            # Upload final model
            final_artifact = wandb.Artifact("final_model", type="model")
            final_artifact.add_file(final_model)
            self.run.log_artifact(final_artifact)

            # Upload checkpoints
            for ckpt_file in checkpoint_files:
                ckpt_artifact = wandb.Artifact("checkpoint_model", type="model")
                ckpt_artifact.add_file(ckpt_file)
                self.run.log_artifact(ckpt_artifact)
            # Upload config
            hydra_artifact = wandb.Artifact("hydra",type="hydra_conf")
            hydra_artifact.add_dir(hydra_dir)
            wandb.log_artifact(hydra_artifact)
            self.run.finish()
