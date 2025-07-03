
import torch.nn as nn
from configs.config import TrainConfig
from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.base_class import BaseAlgorithm
import hydra
from neuronal_networks.other_vision_encoders import PrebuiltResNet
from neuronal_networks.custom_extractor import TMN_Extractor
from utils.printutils import print_model_params
from omegaconf import DictConfig, OmegaConf
import wandb
from wandb.wandb_run import Run
from itertools import chain
from stable_baselines3 import PPO
from neuronal_networks.lr_schedulers import LR_Scheduler

def get_vision_model(cfg : TrainConfig, in_color_channels : int, extractor_out_dim : int) -> nn.Module:
    """Create and return vision model according to configuration"""
    vision_model_constructor = hydra.utils.instantiate(cfg.models)

    #this may not be pretty, but not having the correct input channels has caused headaches
    expected_inchannel = 1 if cfg.rl_env.obsmanager.colorspace == "grayscale" else 3
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
  
    policy_kwargs = dict(
    features_extractor_class=TMN_Extractor,
    features_extractor_kwargs= dict( 
    vision_model = vision_model,
    device= device,
    out_dim =  cfg.extractors_out_dim)
    )
    algorithm_params = OmegaConf.to_container(cfg.sb3.algorithm_params, resolve=True)

    if not (load_model_path is None):
        # TODO remove the PPO and make this modular so it can be used with different algos
        model = PPO(policy = "MultiInputPolicy",env= tm_env, policy_kwargs=policy_kwargs,tensorboard_log= run_id,device=device ,**algorithm_params)
        model.set_parameters(load_model_path)
        """ TODO : this didnt work on windows but apparently on linux???
        model = PPO.load(path = load_model_path,env=tm_env,custom_objects={
            "features_extractor_class": TMN_Extractor,
            "features_extractor_kwargs": policy_kwargs["features_extractor_kwargs"]
        })"""
    else:
        lr : LR_Scheduler = hydra.utils.instantiate(cfg.lr_scheduler)
        model_constructor = hydra.utils.instantiate(cfg.sb3.constructor)
        model : BaseAlgorithm | PPO | SAC | DQN = model_constructor(env= tm_env,learning_rate = lr.get_scheduler(), policy_kwargs=policy_kwargs,tensorboard_log= run_id,device=device ,**algorithm_params)
    
    if print_params:
        print("\nExtractor, Policy and Critic architecturs:\n" + "-"*30)
        print_model_params(model) 
        if True:
            print("\nFeature Extractor Parameters:\n" + "-"*30)
            for name, param in chain(model.policy.features_extractor.named_parameters(),model.policy.mlp_extractor.named_parameters()):
                print(f"{name}: requires_grad = {param.requires_grad}")
            print("\nActor- and Value-Networks Parameters:\n" + "-"*30)
            for name, param in chain(model.policy.action_net.named_parameters(),model.policy.value_net.named_parameters()):
                print(f"{name}: requires_grad = {param.requires_grad}")

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