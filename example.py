"""
This is an empty template showing basic interactions with a TMI2 interface with Python.
"""
import argparse

#from trackmania_rl.tmi_interaction.tminterface2 import MessageType, TMInterface
from tminterface2 import MessageType, TMInterface

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmi_port", "-p", type=int, default=8775)
    args = parser.parse_args()
    iface = TMInterface(args.tmi_port)

    if not iface.registered:
        while True:
            try:
                iface.register(10)
                break
            except ConnectionRefusedError as e:
                print(e)

    
    
    while True:
        msgtype = iface._read_int32()
        # =============================================
        #        READ INCOMING MESSAGES
        # =============================================
        if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
            print("SC_RUN_STEP_SYNC")
            _time = iface._read_int32()
            # ============================
            # BEGIN ON RUN STEP
            # ============================

            # ============================
            # END ON RUN STEP
            # ============================
            iface._respond_to_call(msgtype)
        elif msgtype == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
            print("SC_CHECKPOINT_COUNT_CHANGED_SYNC")
            current = iface._read_int32()
            target = iface._read_int32()
            # ============================
            # BEGIN ON CP COUNT
            # ============================
            # ============================
            # END ON CP COUNT
            # ============================
            iface._respond_to_call(msgtype)
        elif msgtype == int(MessageType.SC_LAP_COUNT_CHANGED_SYNC):
            print(iface._read_int32())
            iface._respond_to_call(msgtype)
        elif msgtype == int(MessageType.SC_REQUESTED_FRAME_SYNC):
            print("Requrested frame synch")
            iface._respond_to_call(msgtype)
        elif msgtype == int(MessageType.C_SHUTDOWN):
            print("Shutdown")
            iface.close()
        elif msgtype == int(MessageType.SC_ON_CONNECT_SYNC):
            print("SC_ON_CONNECT_SYNC")
            iface._respond_to_call(msgtype)
        else:
            print("Setting gas to -50000")
            iface.execute_command("gas -50000")
            pass
    

if __name__ == "__main__":
    main()