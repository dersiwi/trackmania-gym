# utils
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # TODO : <- i don't want this here and it shouldnt have to be here!!!

from utils.scriptargs import get_argparser
from pathlib import Path
from contextlib import redirect_stdout
from multiprocessing import Queue, Process
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym
import numpy as np

# imports for communication between TMInterface and environment
from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
from game_interaction.game_instance_manager2 import GameInstanceManager
from game_interaction.run_multiprocess_wrapper import run_wrapper, start_process_and_wait_for_startsignal
from game_interaction.process_wrapper import TMIProcessWrapper

# gymnasium environment wrapper 
from gymnasium import ObservationWrapper
from trackmania_env.wrappers.observations_filter import ObservationFilter
from trackmania_env.wrappers.bgra_to_rgb import BGRA_to_RGB
from trackmania_env.wrappers.transform_grayscale import TransformGrayscale
from trackmania_env.wrappers.transform_torch import PytorchWrapper
from trackmania_env.wrappers.observations_filter import ObservationFilter


# extractor imports
from neuronal_networks.conv_NNs import PrebuiltResNet
from neuronal_networks.custom_extractor import TMN_Extractor
import torch.nn as nn

from simstate_space_dict import simstate_space_dict
from configs.config import TrainConfig
from utils.printutils import print_model_params



_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "../configs",
    "config_name": "train.yaml",
}

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

@hydra.main(**_HYDRA_PARAMS)
def main(cfg:TrainConfig):
    
    # Instanciate GMI, TMNF-Environment and start TMi-Interaction process.
    platform = cfg.platforms

    GIM = GameInstanceManager.get_instance(
        TMLoader_path = platform.tmloader,
        path_to_plugin = platform.plugin,
        TMLoader_profile_name= cfg.gmi.tm_loader_profile_name,
        linux = platform.os == "linux",
        headless= cfg.gmi.headless)

    tmi_process, control_queue, response_queue = start_process_and_wait_for_startsignal(GIM, cfg.gmi.launch, cfg.image.width, cfg.image.height)

    tm_env = TMNF_Single_Agent_Env(
        command_queue=control_queue,
        response_queue=response_queue)
    
    # apply (Observation)-wrappers to the environment
    for _, wrapper_conf in cfg.rl_env.wrappers.items():
        wrapper : ObservationWrapper = hydra.utils.instantiate(wrapper_conf)
        tm_env = wrapper(env=tm_env)
    
    # get algorithm and start learning process
    vision_model, model = get_models(cfg, tm_env, print_params = True)
    model.learn(cfg.total_timesteps)


    # Finalize training and close game all processes.
    control_queue.put(TMIProcessWrapper.IPCCommands.get_end_syncloop_command(1000)) #1000 doesnt matter.
    tmi_process.join()
        
    if cfg.gmi.launch:
        GIM.close_game()


if __name__ == "__main__": 
    main()
    