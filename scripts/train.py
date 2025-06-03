# utils
import sys, os
# TODO : <- i don't want this here and it shouldnt have to be here!!!
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

# Hydra related imports
import hydra
import omegaconf

# Weights and Biases related imports
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback

# imports for communication between TMInterface and environment
from game_interaction.game_instance_manager2 import GameInstanceManager
from game_interaction.run_multiprocess_wrapper import run_wrapper
from game_interaction.process_wrapper import TMIProcessWrapper

# gymnasium environment wrapper 
from gymnasium import ObservationWrapper
from stable_baselines3.common.monitor import Monitor

# extractor imports
from neuronal_networks.conv_NNs import PrebuiltResNet
from neuronal_networks.custom_extractor import TMN_Extractor
import torch.nn as nn
from stable_baselines3.common.base_class import BaseAlgorithm
from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
from configs.config import TrainConfig
from utils.printutils import print_model_params

# unused imports
from simstate_space_dict import simstate_space_dict
from utils.scriptargs import get_argparser
from pathlib import Path
from contextlib import redirect_stdout
from multiprocessing import Queue, Process
import logging
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym
from stable_baselines3.common.policies import ActorCriticPolicy
import numpy as np

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "../configs",
    "config_name": "train.yaml",
}

def get_models(cfg : TrainConfig, run_id:str ,tm_env : TMNF_Single_Agent_Env, print_params : bool = False) -> tuple[nn.Module, BaseAlgorithm]:

    """instanciates vision-model as well as sb3 algorithm according to parameters
    
    - cfg : config containing global configuration
    - run_id: identifier for the run 
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

    model_constructor = hydra.utils.instantiate(cfg.sb3.constructor)
    model : BaseAlgorithm = model_constructor(env= tm_env, policy_kwargs=policy_kwargs,tensorboard_log= f"runs/{run_id}")

    if print_params:
        print_model_params(model)       

    return vision_model, model

@hydra.main(**_HYDRA_PARAMS)
def main(cfg:TrainConfig):

    # Start Weights and Biases login
    run_id = ""
    if cfg.wandb.use :
        wandb.login()
        wandb.config = omegaconf.OmegaConf.to_container(cfg, resolve=True,throw_on_missing=True)
        run = wandb.init(
            entity=cfg.wandb.entity, 
            project=cfg.wandb.project,
            sync_tensorboard=True, 
            monitor_gym=True,  
            save_code=True)
        run_id = run.id
        
    # Instanciate GMI, TMNF-Environment and start TMi-Interaction process.
    platform = cfg.platforms
    #args = get_argparser()
    #config_logging()
    #logger = logging.getLogger("examplescript")

    IMG_WIDTH, IMG_HEIHGT = cfg.image.width, cfg.image.height

    GIM = GameInstanceManager.get_instance(
        TMLoader_path = platform.tmloader,
        path_to_plugin = platform.plugin,
        TMLoader_profile_name= cfg.gmi.tm_loader_profile_name,
        linux = platform.os == "linux",
        headless= cfg.gmi.headless)

    # instanciate a SyncManager.
    control_queue = Queue() # queue for commands to send to TMIProcessWrapper
    response_queue = Queue() # answers (payload) from TMIProcess Wrapper
   
    p = Process(target=run_wrapper, args=(GIM, cfg.gmi.launch, control_queue, response_queue, IMG_WIDTH, IMG_HEIHGT))
    
    p.start()

    # wait for trackmania to load map and start simulation.
    control_queue.put_nowait(TMIProcessWrapper.IPCCommands.get_startsignal(512))
    startsignal = response_queue.get(timeout = 60)
    assert startsignal["cmd_id"] == 512 and startsignal["status"] == 0

    #gym.make_vec("TMNF_Single_Agent_ENV_v0",num_envs=2,)
    tm_env = TMNF_Single_Agent_Env(
        command_queue=control_queue,
        response_queue=response_queue)
    #tm_env = Monitor(tm_env)

    # apply (Observation)-wrappers to the environment
    for _, wrapper_conf in cfg.rl_env.wrappers.items():
        wrapper = hydra.utils.instantiate(wrapper_conf)
        tm_env = wrapper(tm_env)
    
    # get algorithm, start learning process and initialize weights and biases callback for tracking metrics if needed
    vision_model, model = get_models(cfg, run_id ,tm_env, print_params = True)
    # for sb3 the type would be BaseCallBack. For other callbacks we would need to manually write the other types.
    # TODO check if callbacks have the same class they inherit from 
    callback = hydra.utils.instantiate(cfg.wandb_callbacks)(model_save_path=f"models/{run_id}")  if cfg.wandb.use else None  
    model.learn(**cfg.sb3.learn_args,callback=callback)


    # Finalize training and close game all processes.
    control_queue.put(TMIProcessWrapper.IPCCommands.get_end_syncloop_command(1000)) #1000 doesnt matter.
    p.join()
        
   
    GIM.close_game()


if __name__ == "__main__": 
    main()
    