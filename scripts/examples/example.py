"""
This is an empty template showing basic interactions with a TMI2 interface with Python.
The template registers an TMInterface with 10 seconds of timeout.

Afterwards, it communicates with the game via a binary protocol; functions defined in TMInterface.
The first message from the server is always the type of message, while the second message is the payload. 
Each message has to be acknowledged by the client.
"""
import argparse
import sys, os
from pathlib import Path
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # TODO : <- i don't want this here and it shouldnt have to be here!!!


from game_interaction.tminterface2 import MessageType, TMInterface
from game_interaction.game_instance_manager2 import GameInstanceManager
from utils.print_simstate import print_sim_state
from utils.scriptargs import get_paths, get_argparser, config_logging


"""
used for width and height of image, if rqimgs is set
    -IMG_W, IMG_H : width and height of image
    -SAVE_FREQUENCY : save image every xth simulation step
"""
IMG_W = 160
IMG_H = 120
SAVE_FREQUENCY = 10 

"""
define a set of inputs to the game, like :  left: bool, right: bool, accelerate: bool, brake: bool
    - If inputset is None, no inputs are sent.
    - INPUT_SET_FREQUENCY : sends inputs every xth simulation step
    - REPEAT : repeats the inputs from beginning once all inputs have been sent, if not, ends.
"""
set1 = [(False, False, True, False)]
inputset = None #set1
INPUT_SET_FREQUENCY = 2
REPEAT = True

"""Gets simulation state in specified frequency and prints it to the console"""
STATE_FREQUENCY = 10





def main(iface : TMInterface, rqimgs : bool = False):

    frame_id = 0
    stepcount = 0 
    inputcounter = 0
    while True:
        msgtype = iface._read_int32()
        
        # ============================================= READ INCOMING MESSAGES
        
        
        if msgtype == int(MessageType.SC_RUN_STEP_SYNC): # simulation step is complete

            _time = iface._read_int32() # _time in this case is the total simulation time (i think)

            # ============================ BEGIN ON RUN STEP ============================



            if rqimgs and stepcount % SAVE_FREQUENCY == 0:
                iface.request_frame(IMG_W, IMG_H)
            

            if not inputset is None and inputcounter < len(inputset) and stepcount % INPUT_SET_FREQUENCY == 0:
                # server should not reply to this command.
                (left, right, acc, brake) = inputset[inputcounter]
                iface.set_input_state(left, right, acc, brake)
                inputcounter += 1
                if inputcounter >= len(inputset) and REPEAT:
                    inputcounter = 0

            if stepcount % STATE_FREQUENCY == 0:
                ssD = iface.get_simulation_state()
                print_sim_state(ssD)
                


            # ============================ END ON RUN STEP ============================
            iface._respond_to_call(msgtype)


        elif msgtype == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):

            current = iface._read_int32()
            target = iface._read_int32()

            # ============================ BEGIN ON CP COUNT ============================

            # ============================ END ON CP COUNT ============================
            iface._respond_to_call(msgtype)
        elif msgtype == int(MessageType.SC_LAP_COUNT_CHANGED_SYNC):

            iface._respond_to_call(msgtype)

        elif msgtype == int(MessageType.SC_REQUESTED_FRAME_SYNC):

            frame = iface.get_frame(IMG_W, IMG_H)
            np.save(os.path.join("frames", f"frameframe_{frame_id}"), frame)
            frame_id += 1
            iface._respond_to_call(msgtype)

        elif msgtype == int(MessageType.C_SHUTDOWN):

            iface.close()

        elif msgtype == int(MessageType.SC_ON_CONNECT_SYNC):
            print("--------------------On connect event.!--------------------------")
            iface.on_connect_event()
            iface._respond_to_call(msgtype)
        else:
            iface._respond_to_call(msgtype)
            pass

        stepcount += 1
    

if __name__ == "__main__":
    args = get_argparser
    tmloader, plugin = get_paths(args.linux)

    GMI = GameInstanceManager.get_instance(TMLoader_path = tmloader,
                                path_to_plugin = plugin,
                                linux = args.linux)
        
    if args.launch:
        GMI.launch_game()

    GMI.register_iface()
    iface = GMI.get_tminterface()


    try:
        main(iface)
    except TimeoutError as e:
        print(e)

    if args.launch:
        GMI.close_game()