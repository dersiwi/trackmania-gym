import sys, os
from pathlib import Path
import gymnasium as gym
from contextlib import redirect_stdout
import numpy as np
from multiprocessing import Queue, Process
import time 
import logging


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # TODO : <- i don't want this here and it shouldnt have to be here!!!


from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
from trackmania_env.envs.actionmap import get_reverse_action_map
from trackmania_env.wrappers.observations_filter import ObservationFilter


from game_interaction.game_instance_manager2 import GameInstanceManager
from game_interaction.run_multiprocess_wrapper import run_wrapper
from game_interaction.process_wrapper import TMIProcessWrapper
from game_interaction.tminterface_commands import TMInterfaceCommands


from simstate_space_dict import simstate_space_dict
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
        img_width=IMG_WIDTH,
        img_height=IMG_HEIHGT,
        command_queue=control_queue,
        response_queue=response_queue)
    
    # TODO wrap here the env with observation filter wrapper
    observations_list = [
            "image",
            "position",
            "velocity",
            "yaw_pitch_roll",
            "scene_mobil.sync_vehicle_state.speed_forward",
            "scene_mobil.input_steer",
            "scene_mobil.async_vehicle_state.is_turbo",
            "scene_mobil.engine.gear",
            "scene_mobil.async_vehicle_state.rest",
            "player_info.last",
            ]
    # stress test to make sure all fields work
    observations_list = list(simstate_space_dict.keys())
    tm_env = ObservationFilter(tm_env, observations_list)
    
    obs: gym.spaces.Dict = tm_env._get_obs()
    with open('out.txt', 'w') as f:
        with redirect_stdout(f):
            for i , (k,v) in enumerate(obs.items()):
                print(k)
                print(v)
                print("-"*20)

    if args.replay:
        logger.info(f"Replaying actions from {args.replay}")
        with open(args.replay, "r") as file:
            actions = [eval(line.strip()) for line in file][0]

        RAM = get_reverse_action_map()
        for action in actions:
            tm_env.step(RAM[action])

    else:
        for i in range(100):
            tm_env.step(np.random.randint(0, 12))
        original_env : TMNF_Single_Agent_Env = tm_env.env
        original_env.store_actions("logs/action_log.txt")
    
    control_queue.put(TMIProcessWrapper.IPCCommands.get_cmd_command(999, TMInterfaceCommands.recover_inputs("inputs.txt")))

    
    
    control_queue.put(TMIProcessWrapper.IPCCommands.get_end_syncloop_command(1000)) #1000 doesnt matter.
    p.join()