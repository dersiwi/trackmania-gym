"""
This is an empty template showing basic interactions with a TMI2 interface with Python.
The template registers an TMInterface with 10 seconds of timeout.

Afterwards, it communicates with the game via a binary protocol; functions defined in TMInterface.
The first message from the server is always the type of message, while the second message is the payload. 
Each message has to be acknowledged by the client.
"""
import argparse
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # TODO : <- i don't want this here and it shouldnt have to be here!!!

#from trackmania_rl.tmi_interaction.tminterface2 import MessageType, TMInterface
from game_interaction.tminterface2 import MessageType, TMInterface
from game_interaction.game_instance_manager2 import GameInstanceManager
from pathlib import Path
import numpy as np
from tminterface.structs import CheckpointData, SimStateData, CheckpointTime

from bytefield import ArrayField
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

def print_sim_state(ssD : SimStateData) -> None:
    print("\n=== Simulation State Snapshot ===")
    print(f"    Time:                {ssD.time} ms")
    print(f"    Position:            {ssD.position}")
    print(f"    Velocity:            {ssD.velocity}")
    print(f"    Rotation (YawPitchRoll): {ssD.yaw_pitch_roll}")
    print(f"    Display Speed:       {ssD.display_speed} units")
    
    print("\nInput State:")
    print(f"    Accelerate:          {ssD.input_accelerate}")
    print(f"    Brake:               {ssD.input_brake}")
    print(f"    Left:                {ssD.input_left}")
    print(f"    Right:               {ssD.input_right}")
    print(f"    Steer (analog):      {ssD.input_steer}")
    print(f"    Gas (analog):        {ssD.input_gas}")
    
    print("\nRace Progress:")
    print(f"    Race Time:           {ssD.race_time} ms")
    print(f"    Rewind Time:         {ssD.rewind_time} ms")
    print(f"    Num Respawns:        {ssD.num_respawns}")

    cpData : CheckpointData = ssD.cp_data
    cp_times_structs : ArrayField[CheckpointTime] = cpData.cp_times  # This is likely a list of structs

    cp_times = [cp_times_structs[i].time for i in range(cpData.cp_times_length)]
    print(f"   Checkpoint Times:    {cp_times}")
    print(f"    Checkpoints Passed:  {cpData.cp_states_length}")



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
        main(iface)
    except TimeoutError as e:
        print(e)

    if args.launch:
        GMI.close_game()