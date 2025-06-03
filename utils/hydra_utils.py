
import torch.nn as nn
from configs.config import TrainConfig
from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
import hydra
from neuronal_networks.conv_NNs import PrebuiltResNet
from neuronal_networks.custom_extractor import TMN_Extractor
from utils.printutils import print_model_params



def get_models(cfg : TrainConfig, tm_env : TMNF_Single_Agent_Env, print_params : bool = False) -> tuple[nn.Module, BaseAlgorithm | PPO]:

    """instanciates vision-model as well as sb3 algorithm according to parameters
    
    - cfg : config containing global configuration 
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
    vision_model = vision_model,
    vision_model_out_dim = vision_model.out_dims))

    model_constructor = hydra.utils.instantiate(cfg.sb3)
    model : BaseAlgorithm | PPO = model_constructor(env= tm_env, policy_kwargs=policy_kwargs)

    if print_params:
        print_model_params(model)       

    return vision_model, model