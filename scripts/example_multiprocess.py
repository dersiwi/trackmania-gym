
import argparse
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # TODO : <- i don't want this here and it shouldnt have to be here!!!

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


def run_wrapper(iface, cmd_q, res_q, img_w, img_h): # apparently its better to run process like this to avoid pickel issues or smth?
    wrapper = TMIProcessWrapper(iface, command_queue=cmd_q, response_queue=res_q, img_width=img_w, img_height=img_h)
    wrapper.syncloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmi_port", "-p", type=int, default=8775)
    parser.add_argument("--launch", "-l", action="store_true",  default=False)
    parser.add_argument("--linux", "-u", action="store_true",  default=False)
    parser.add_argument("--reqimgs", "-imgs", action="store_true",  help="If set, requests images each simulation step and stores them in the current directory in a /frame folder. [WARNING : This is a ton of frames, even for short amounts of running it.]", default=False)

    args = parser.parse_args()

    if args.linux:
        tmloader = None
        plugin = None
    else:
        tmloader = Path(os.path.expanduser("~")) / "AppData" / "Local" / "TMLoader" / "TMLoader.exe"
        plugin = Path(os.path.expanduser("~")) / "OneDrive" / "Dokumente" / "TMInterface" /"Plugins" / "Python_Link.as"



    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/app.log"),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger("examplescript")

    

    GMI = GameInstanceManager.get_instance(TMLoader_path = tmloader,
                            path_to_plugin = plugin,
                            linux = args.linux)
    
    if args.launch:
        GMI.launch_game()

    GMI.register_iface()
    iface = GMI.get_tminterface()

    # instanciate a SyncManager.
    control_queue = Queue() # queue for commands to send to TMIProcessWrapper
    response_queue = Queue() # answers (payload) from TMIProcess Wrapper

    p = Process(target=run_wrapper, args=(iface, control_queue, response_queue, 180, 160))
    
    p.start()

    #interact with process here
    cmd_id = 1
    try:
        print("Putting img request command")
        control_queue.put_nowait(TMIProcessWrapper.IPCCommands.get_req_img_command(cmd_id))
        imgpack = response_queue.get(timeout=10)
        print("Received image.")
        print(imgpack["sim_state"])
        print(imgpack["sim_step"])
        print(imgpack["status"])
        print(imgpack["cmd_id"])

        control_queue.put_nowait(TMIProcessWrapper.IPCCommands.get_end_syncloop_command(cmd_id + 1))
    except Full:
        print("Queue is full.")



    p.join()
    

    if args.launch:
        GMI.close_game()