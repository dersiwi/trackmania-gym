"""
This is an empty template showing basic interactions with a TMI2 interface with Python.
The template registers an TMInterface with 10 seconds of timeout.

Afterwards, it communicates with the game via a binary protocol; functions defined in TMInterface.
The first message from the server is always the type of message, while the second message is the payload. 
Each message has to be acknowledged by the client.
"""
import argparse

#from trackmania_rl.tmi_interaction.tminterface2 import MessageType, TMInterface
from tminterface2 import MessageType, TMInterface
from game_instance_manager2 import GameInstanceManager
from pathlib import Path
import os
import numpy as np

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
inputset = set1
INPUT_SET_FREQUENCY = 2
REPEAT = True


def main(tmi_port : int, rqimgs : bool = False):

    iface = TMInterface(tmi_port)

    if not iface.registered:
        while True:
            try:
                iface.register(10)
                break
            except ConnectionRefusedError as e:
                print(e)

    
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

            iface._respond_to_call(msgtype)

        else:
            iface._respond_to_call(msgtype)
            pass

        stepcount += 1
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmi_port", "-p", type=int, default=8775)
    parser.add_argument("--launch", "-l", action="store_true",  default=False)
    parser.add_argument("--reqimgs", "-imgs", action="store_true",  help="If set, requests images each simulation step and stores them in the current directory in a /frame folder. [WARNING : This is a ton of frames, even for short amounts of running it.]", default=False)

    args = parser.parse_args()

    if args.launch:
        GMI = GameInstanceManager(Path(os.path.expanduser("~")) / "AppData" / "Local" / "TMLoader" / "TMLoader.exe",
                                "default",
                                Path(os.path.expanduser("~")) / "OneDrive" / "Dokumente" / "TMInterface" /"Plugins" / "Python_Link.as")
        GMI.launch_game()
    main(args.tmi_port)

    if args.launch:
        GMI.close_game()