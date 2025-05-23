from multiprocessing import Process

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

from bytefield import ArrayField

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

    GMI = GameInstanceManager.get_instance(TMLoader_path = tmloader,
                            path_to_plugin = plugin,
                            linux = args.linux)
    
    if args.launch:
        GMI.launch_game()

    GMI.register_iface()
    iface = GMI.get_tminterface()


    try:
        tmiprocess = TMIProcessWrapper(iface, 180, 160, img_req_frequency=3)
        p = Process(target=tmiprocess.syncloop)
        p.start()
        for i in range(10):
            idx = tmiprocess.request_image()
            img = tmiprocess.get_imgage_blocking(idx)
            print("Got an image!")
            
            
        p.join()
        
    except TimeoutError as e:
        print(e)

    if args.launch:
        GMI.close_game()