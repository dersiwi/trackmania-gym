
import torch.nn as nn
from configs.config import TrainConfig
from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
import hydra
from neuronal_networks.conv_NNs import PrebuiltResNet
from neuronal_networks.custom_extractor import TMN_Extractor
from utils.printutils import print_model_params
from omegaconf import DictConfig, OmegaConf
import wandb
from wandb.wandb_run import Run

def get_models(cfg : TrainConfig, tm_env : TMNF_Single_Agent_Env, print_params : bool = False,run_id:str = "test") -> tuple[nn.Module, BaseAlgorithm | PPO]:

    """instanciates vision-model as well as sb3 algorithm according to parameters
    
    - cfg : config containing global configuration 
    - run_id: identifier for the run which gets used for tensorboard login
    - tm_env : gym-environment for algorithm
    - print_params : If True, prints shapes of weights of neural network

    Basically it does this : model = PPO("MultiInputPolicy", env= tm_env, policy_kwargs=policy_kwargs, verbose=1), 
    in a fancy way.

    Returns vision model as well as the algorithm."""

    vision_model_constructor = hydra.utils.instantiate(cfg.models)
    vision_model : nn.Module | PrebuiltResNet = vision_model_constructor(in_color_channels=tm_env.observation_space["image"].shape[-1])
  
    policy_kwargs = dict(
    features_extractor_class=TMN_Extractor,
    features_extractor_kwargs= dict( 
    vision_model = vision_model))#,
    #vision_model_out_dim = vision_model.out_dims))

    algorithm_params = OmegaConf.to_container(cfg.sb3.algorithm_params, resolve=True)



    model_constructor = hydra.utils.instantiate(cfg.sb3.constructor)
    model : BaseAlgorithm | PPO | SAC = model_constructor(env= tm_env, policy_kwargs=policy_kwargs,tensorboard_log= run_id, **algorithm_params)

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