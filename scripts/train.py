# utils
import sys, os

import omegaconf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # TODO : <- i don't want this here and it shouldnt have to be here!!!
from utils.scriptargs import get_argparser
from pathlib import Path
from contextlib import redirect_stdout
from multiprocessing import Queue, Process

# imports for communication between TMInterface and environment
from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
from game_interaction.game_instance_manager2 import GameInstanceManager
from game_interaction.run_multiprocess_wrapper import run_wrapper
from game_interaction.process_wrapper import TMIProcessWrapper

# gymnasium environment wrapper 
from trackmania_env.wrappers.observations_filter import ObservationFilter
from trackmania_env.wrappers.bgra_to_rgb import BGRA_to_RGB
from trackmania_env.wrappers.transform_grayscale import TransformGrayscale
from trackmania_env.wrappers.transform_torch import PytorchWrapper

from trackmania_env.wrappers.observations_filter import ObservationFilter
# extractor imports
from neuronal_networks.conv_NNs import PrebuiltResNet
from neuronal_networks.custom_extractor import TMN_Extractor


from simstate_space_dict import simstate_space_dict

import gymnasium as gym
import numpy as np
import logging


from utils.scriptargs import get_argparser, get_paths, config_logging

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import hydra
from omegaconf import DictConfig,OmegaConf

import wandb
from wandb.integration.sb3 import WandbCallback

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "../configs",
    "config_name": "train.yaml",
}

@hydra.main(**_HYDRA_PARAMS)
def main(cfg:DictConfig):
    if True:
        wandb.login()
        wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
        )
        run = wandb.init(entity=cfg.wandb.entity, 
               project=cfg.wandb.project,
                sync_tensorboard=True, 
                monitor_gym=True,  
                save_code=True)
    
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
    #observations_list= cfg.rl_env.observations_list
    #observations_list = list(simstate_space_dict.keys())
    for _,wrapper_conf in cfg.rl_env.wrappers.items():
        wrapper = hydra.utils.instantiate(wrapper_conf)
        tm_env = wrapper(env=tm_env)
    
    vision_model = hydra.utils.instantiate(cfg.models)(in_color_channels=tm_env.observation_space["image"].shape[-1])
  
    policy_kwargs = dict(
    features_extractor_class=TMN_Extractor,
    features_extractor_kwargs= dict( 
    vision_model = vision_model,
    vision_model_out_dim = vision_model.out_dims))

    model = hydra.utils.instantiate(cfg.sb3)(env= tm_env, policy_kwargs=policy_kwargs,tensorboard_log= f"runs/{run.id}")
    print("Number of learnable params: ",len(list(model.policy.named_parameters())))
    #model = PPO("MultiInputPolicy", env= tm_env, policy_kwargs=policy_kwargs, verbose=1)
            
    model.learn(total_timesteps =cfg.total_timesteps,progress_bar=True,
                callback=WandbCallback(gradient_save_freq=100,model_save_path=f"models/{run.id}",verbose=2,),)

    control_queue.put(TMIProcessWrapper.IPCCommands.get_end_syncloop_command(1000)) #1000 doesnt matter.
    p.join()
        
   
    GIM.close_game()


if __name__ == "__main__": 
    main()
    