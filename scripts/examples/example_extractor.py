# utils
import sys, os
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

# extractor imports
from neuronal_networks.conv_NNs import PrebuiltResNet
from neuronal_networks.custom_extractor import TMN_Extractor

from simstate_space_dict import simstate_space_dict

import gymnasium as gym
import numpy as np
import logging


from utils.scriptargs import get_argparser, get_paths, config_logging

if __name__ == "__main__":
    args = get_argparser()
    TMLoader_path, path_to_plugin = get_paths(args.linux)
    config_logging()
    logger = logging.getLogger("examplescript")


    IMG_WIDTH, IMG_HEIHGT = 180, 160

    GIM = GameInstanceManager.get_instance(
        TMLoader_path = TMLoader_path,
        path_to_plugin = path_to_plugin,
        linux = args.linux,
        headless= False)

    # instanciate a SyncManager.
    control_queue = Queue() # queue for commands to send to TMIProcessWrapper
    response_queue = Queue() # answers (payload) from TMIProcess Wrapper
   

    p = Process(target=run_wrapper, args=(GIM, args.launch, control_queue, response_queue, IMG_WIDTH, IMG_HEIHGT))
    
    p.start()


    # wait for trackmania to load map and start simulation.
    control_queue.put_nowait(TMIProcessWrapper.IPCCommands.get_startsignal(512))
    startsignal = response_queue.get(timeout = 60)
    assert startsignal["cmd_id"] == 512 and startsignal["status"] == 0

    tm_env = TMNF_Single_Agent_Env(
        command_queue=control_queue,
        response_queue=response_queue)
    
    observations_list= [
            "image",
            "position",
            "velocity",
            "yaw_pitch_roll",
            "scene_mobil.sync_vehicle_state.speed_forward",
            "scene_mobil.input_steer",
            "scene_mobil.async_vehicle_state.is_turbo",
            "scene_mobil.engine.gear",
            "input_finish_event.input_data_field",
            "input_brake_event.last_field",
            "input_accelerate_event.time_field",
            "scene_mobil.async_vehicle_state.rest_field",
            "player_info.last_field",
            ]
    #observations_list = list(simstate_space_dict.keys())
    tm_env = ObservationFilter(tm_env,observations_list)
    tm_env = BGRA_to_RGB(tm_env)
    tm_env = TransformGrayscale(tm_env,keep_dim=True)
    tm_env = PytorchWrapper(tm_env)

    out_dim = 10
    vision_model = PrebuiltResNet("resnet18",out_dims=out_dim,pretrained=True,
                                  in_color_channels=tm_env.observation_space["image"].shape[-1])
    extractor = TMN_Extractor(tm_env.observation_space,vision_model=vision_model,vision_model_out_dim=out_dim)
    with open('example_extractor_out.txt', 'w') as f:
        with redirect_stdout(f):
             for i in range(20):
                obs,_,_,_,_ = tm_env.step(np.random.randint(0,12))
                v = extractor(obs)
                print(v)
                print(v.shape)
                print("-"*20)
   
    control_queue.put(TMIProcessWrapper.IPCCommands.get_end_syncloop_command(1000)) #1000 doesnt matter.
    p.join()
        
    if args.launch:
        GIM.close_game()
