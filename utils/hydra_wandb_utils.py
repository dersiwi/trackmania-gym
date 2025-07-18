
import torch.nn as nn
from configs.config import TrainConfig
from neuronal_networks.lr_schedulers import LR_Scheduler
from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.base_class import BaseAlgorithm
import hydra
from neuronal_networks.other_vision_encoders import PrebuiltResNet
from neuronal_networks.custom_extractor import TMN_Extractor
from omegaconf import DictConfig, OmegaConf
import wandb
from wandb.wandb_run import Run
from itertools import chain
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy,BasePolicy
from neuronal_networks.get_policy import get_policy

import os

def print_model_params(model : BaseAlgorithm):
    """"Prints parametrs of the given model"""
    print("\nExtractor, Policy and Critic architecturs:\n" + "-"*30)

    for name, param in model.policy.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")

    print("\nFeature Extractor Parameters:\n" + "-"*30)
    for name, param in chain(model.policy.features_extractor.named_parameters(),model.policy.mlp_extractor.named_parameters()):
        print(f"{name}: requires_grad = {param.requires_grad}")
    print("\nActor- and Value-Networks Parameters:\n" + "-"*30)
    for name, param in chain(model.policy.action_net.named_parameters(),model.policy.value_net.named_parameters()):
        print(f"{name}: requires_grad = {param.requires_grad}")
    
    if isinstance(model.policy, ActorCriticPolicy):
        print("\n[INFO] Checking whether the actor and critic are using the same feature extractor:\n" + "-"*30)
        actor_id = id(model.policy.pi_features_extractor)
        critic_id = id(model.policy.vf_features_extractor)

        print("Actor Feature Extractor ID:", actor_id)
        print("Critic Feature Extractor ID:", critic_id)

        if actor_id != critic_id:
            print("Actor and Critic are using DIFFERENT feature extractors.")
        else:
            print("Actor and Critic are sharing the SAME feature extractor.")



def get_vision_model(cfg : TrainConfig, in_color_channels : int, extractor_out_dim : int) -> nn.Module:
    """Create and return vision model according to configuration"""
    vision_model_constructor = hydra.utils.instantiate(cfg.models)

    #this may not be pretty, but not having the correct input channels has caused headaches
    expected_inchannel = 1 if cfg.rl_env.obs_manager.colorspace == "grayscale" else 3
    assert in_color_channels == expected_inchannel, f"Expected {expected_inchannel} color channels, got {in_color_channels}"

    vision_model : nn.Module | PrebuiltResNet = vision_model_constructor(
        in_color_channels=in_color_channels,
        out_dim = extractor_out_dim)
    return vision_model

def get_models(cfg : TrainConfig, tm_env : TMNF_Single_Agent_Env, print_params : bool = False,run_id:str = "test",load_model_path: str | None = None) -> tuple[nn.Module, BaseAlgorithm | PPO]:

    """instanciates vision-model as well as sb3 algorithm according to parameters
    
    - cfg : config containing global configuration 
    - run_id: identifier for the run which gets used for tensorboard login
    - tm_env : gym-environment for algorithm
    - print_params : If True, prints shapes of weights of neural network

    Basically it does this : model = PPO("MultiInputPolicy", env= tm_env, policy_kwargs=policy_kwargs, verbose=1), 
    in a fancy way.

    Returns vision model as well as the algorithm."""
    device = cfg.platforms.device
    vision_model = get_vision_model(cfg, tm_env.observation_space["image"].shape[0], cfg.extractors_out_dim)
    algorithm_params = OmegaConf.to_container(cfg.sb3.algorithm_params, resolve=True)
    lr : LR_Scheduler = hydra.utils.instantiate(cfg.lr_scheduler)
  
    policy_type, policy_kwargs = get_policy(observation_space=tm_env.observation_space,
                                            action_space=tm_env.action_space,
                                            policy_cfg=cfg.policy, device=device, vision_model=vision_model, learning_rate_scheduler = lr)

    model_args = dict(
    policy = policy_type,
    env = tm_env,
    tensorboard_log = run_id,
    device = device,
    verbose = cfg.policy.verbose,
    **algorithm_params
    )
    # Only include policy_kwargs if they exist
    if policy_kwargs: model_args["policy_kwargs"] = policy_kwargs


    if not (load_model_path is None):
        # TODO remove the PPO and make this modular so it can be used with different algos
        model = PPO(**model_args)
        model.set_parameters(load_model_path)
    else:
        model_constructor = hydra.utils.instantiate(cfg.sb3.constructor)
        model : BaseAlgorithm | PPO | SAC | DQN = model_constructor(**model_args)
    
    if print_params:
        print_model_params(model)

    return vision_model, model


def init_and_login_wandb(cfg : TrainConfig, wandbdir : str = "wandb") -> tuple[Run | None, str]:
    """Instanciates and logs into weights and biases (wandb), if specified in configuration (cfg.wandb.use).
    After login, returns tuple of Run-instance and run-id 
    
    If cfg.wandb.use is False, the returned Run is None."""
    run_id = ""
    if cfg.wandb.use:
        wandb.login()
        wandb_conf = OmegaConf.to_container(cfg, resolve=True,throw_on_missing=True)
        wandb.config = wandb_conf
        run = wandb.init(
            entity=cfg.wandb.entity, 
            project=cfg.wandb.project,
            sync_tensorboard=True, 
            monitor_gym=True,  
            save_code=True,
            dir = wandbdir,
            config=wandb_conf)
        run_id = run.id
        return run, run_id
    else:
        return None, run_id
    

from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from trackmania_env.rewards.reward_calculation import RewardLogCallback, AccumRewardLogCallback
import glob


class BeforeAndAfterTraining:

    def __init__(self, hydra_run_dir : str, cfg : TrainConfig):
        self.hydra_run_dir = hydra_run_dir
        self.cfg = cfg


    def before_training(self) -> None:
        """Create necessary paths and initate wandb-login."""
        model_dir = os.path.join(self.hydra_run_dir, "models")
        self.best_model_path = os.path.join(model_dir, "best_model")
        self.checkpoint_path = os.path.join(model_dir, "checkpoints")
        os.makedirs(self.best_model_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)

        # Start Weights and Biases login
        self.run, run_id = init_and_login_wandb(self.cfg, wandbdir=self.hydra_run_dir)
        self.run_id_in_hydra_log_dir = os.path.join(self.hydra_run_dir, run_id)


    def get_tensorboard_login_identifier(self) -> str:
        """identifier for the run which gets used for tensorboard login"""
        return self.run_id_in_hydra_log_dir


    def get_callbacks_for_training(self, tm_env) -> CallbackList:
        """Create callbacks for model to create logs"""
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
            callback.callbacks.extend([hydra.utils.instantiate(self.cfg.wandb_callbacks)(model_save_path=self.run_id_in_hydra_log_dir), AccumRewardLogCallback()])

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
