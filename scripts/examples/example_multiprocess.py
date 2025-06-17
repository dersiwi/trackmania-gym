
import argparse
import sys, os
sys.path.append(os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__), '..'), '..'))) # TODO : <- i don't want this here and it shouldnt have to be here!!!

#from trackmania_rl.tmi_interaction.tminterface2 import MessageType, TMInterface
from game_interaction.tminterface2 import MessageType, TMInterface
from game_interaction.process_wrapper import TMIProcessWrapper
from game_interaction.game_instance_manager2 import GameInstanceManager
from pathlib import Path
import numpy as np
from tminterface.structs import CheckpointData, SimStateData, CheckpointTime
import time
from multiprocessing import Process, Queue
from queue import Full

from bytefield import ArrayField
import logging
from game_interaction.ipc_fields import IPCFields

from trackmania_env.utils.actionmap import ACTION_MAP
import random

from utils.scriptargs import get_argparser, config_logging, get_paths
from game_interaction.run_multiprocess_wrapper import start_process_and_wait_for_startsignal, run_wrapper

if __name__ == "__main__":
    args = get_argparser()
    tmloader, plugin = get_paths(args.linux)
    config_logging()
    logger = logging.getLogger("examplescript")


    GIM = GameInstanceManager.get_instance(TMLoader_path = tmloader,
                            path_to_plugin = plugin,
                            linux = args.linux)
    

    #GIM.register_iface()           <-- both is not needed as this is done in the process wrapper
    #iface = GIM.get_tminterface()

    launch_game = True
    IMG_WIDTH, IMG_HEIHGT = 100, 100

    # instanciate a SyncManager.
    control_queue = Queue() # queue for commands to send to TMIProcessWrapper
    response_queue = Queue() # answers (payload) from TMIProcess Wrapper

    p = Process(target=run_wrapper, args=(GIM, launch_game, control_queue, response_queue, "simple1_validated.Challenge.Gbx", IMG_WIDTH, IMG_HEIHGT, True))
    
    p.start()

    #need to wait for startup of game
    time.sleep(10)

    #interact with process here
    cmd_id = 1
    try:
        print("Putting img request command")
        control_queue.put_nowait(TMIProcessWrapper.IPCCommands.get_req_img_command(cmd_id))
        imgpack = response_queue.get(timeout=10)
        print("Received image.")
        print(imgpack[IPCFields.SIMSTATE])
        print(imgpack[IPCFields.SIMSTEP])
        print(imgpack[IPCFields.STATUS])
        print(imgpack[IPCFields.CMD_ID])

        cmd_id += 1

        control_queue.put_nowait(TMIProcessWrapper.IPCCommands.get_act_command(cmd_id, ACTION_MAP[0]))
        res = response_queue.get(timeout=10)
        print(res[IPCFields.CMD_ID])

        time.sleep(5)
        cmd_id += 1

        # now send 100 random actions
        for i in range(100):

            control_queue.put_nowait(TMIProcessWrapper.IPCCommands.get_act_command(cmd_id, ACTION_MAP[random.randint(0, len(ACTION_MAP) - 1)]))
            res = response_queue.get(timeout=10)
            print(res[IPCFields.CMD_ID])
            cmd_id += 1

        control_queue.put_nowait(TMIProcessWrapper.IPCCommands.get_end_syncloop_command(cmd_id + 1))
    except Full:
        print("Queue is full.")



    p.join()