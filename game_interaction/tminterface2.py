"""
This module includes code adapted and refactored from the Linesight-AI project
(https://github.com/Linesight-RL/linesight). Credit to the original authors Donadigo @https://github.com/donadigo/TMInterfaceClientPython
for the foundational implementation.
"""

import signal
import socket
import struct
from enum import IntEnum, auto
from sys import platform
import numpy as np
import logging
from tminterface.structs import CheckpointData, SimStateData
# tminterface is pip installed from https://github.com/donadigo/TMInterfaceClientPython

HOST = "127.0.0.1"


class MessageType(IntEnum):
    SC_RUN_STEP_SYNC = auto()
    SC_CHECKPOINT_COUNT_CHANGED_SYNC = auto()
    SC_LAP_COUNT_CHANGED_SYNC = auto()
    SC_REQUESTED_FRAME_SYNC = auto()
    SC_ON_CONNECT_SYNC = auto()
    C_SET_SPEED = auto()
    C_REWIND_TO_STATE = auto()
    C_REWIND_TO_CURRENT_STATE = auto()
    C_GET_SIMULATION_STATE = auto()
    C_SET_INPUT_STATE = auto()
    C_GIVE_UP = auto()
    C_PREVENT_SIMULATION_FINISH = auto()
    C_SHUTDOWN = auto()
    C_EXECUTE_COMMAND = auto()
    C_SET_TIMEOUT = auto()
    C_RACE_FINISHED = auto()
    C_REQUEST_FRAME = auto()
    C_RESET_CAMERA = auto()
    C_SET_ON_STEP_PERIOD = auto()
    C_UNREQUEST_FRAME = auto()
    C_TOGGLE_INTERFACE = auto()
    C_IS_IN_MENUS = auto()
    C_GET_INPUTS = auto()


class TMInterface:
    """https://tminterface.readthedocs.io/en/latest/source/tminterface.html#tminterface.interface.MessageType.S_ON_RUN_STEP"""
    registered = False

    def __init__(self, port: int):
        self.port = port
        self.logger = logging.getLogger(self.__class__.__name__)


    def close(self):
        self.sock.sendall(struct.pack("i", MessageType.C_SHUTDOWN))
        self.sock.close()
        self.registered = False

    def signal_handler(self, sig, frame):
        self.logger.info("Shutting down...")
        self.close()

    def register(self, timeout=None):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        signal.signal(signal.SIGINT, self.signal_handler)
        # https://stackoverflow.com/questions/45864828/msg-waitall-combined-with-so-rcvtimeo
        # https://stackoverflow.com/questions/2719017/how-to-set-timeout-on-pythons-socket-recv-method
        if timeout is not None:
            if platform in ["linux", "linux2"]:  # https://stackoverflow.com/questions/46477448/python-setsockopt-what-is-worng
                timeout_pack = struct.pack("ll", timeout, 0)
            else:
                timeout_pack = struct.pack("q", timeout * 1000)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO, timeout_pack)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDTIMEO, timeout_pack)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((HOST, self.port))
        self.registered = True
        self.logger.info(f"Connected. Timeout set to {timeout}ms.")

    def rewind_to_state(self, state):
        self.sock.sendall(struct.pack("ii", MessageType.C_REWIND_TO_STATE, np.int32(len(state.data))))
        self.sock.sendall(state.data)

    def rewind_to_current_state(self):
        self.sock.sendall(struct.pack("i", MessageType.C_REWIND_TO_CURRENT_STATE))

    def reset_camera(self):
        self.sock.sendall(struct.pack("i", MessageType.C_RESET_CAMERA))

    def get_simulation_state(self) -> SimStateData:
        self.sock.sendall(struct.pack("i", MessageType.C_GET_SIMULATION_STATE))
        state_length = self._read_int32()
        state = SimStateData(self.sock.recv(state_length, socket.MSG_WAITALL))
        state.cp_data.resize(CheckpointData.cp_states_field, state.cp_data.cp_states_length)
        state.cp_data.resize(CheckpointData.cp_times_field, state.cp_data.cp_times_length)
        return state

    def set_input_state(self, left: bool, right: bool, accelerate: bool, brake: bool) -> None:
        self.sock.sendall(
            struct.pack("iBBBB", MessageType.C_SET_INPUT_STATE, np.uint8(left), np.uint8(right), np.uint8(accelerate), np.uint8(brake))
        )

    def give_up(self) -> None:
        self.sock.sendall(struct.pack("i", MessageType.C_GIVE_UP))

    def prevent_simulation_finish(self):
        self.sock.sendall(struct.pack("i", MessageType.C_PREVENT_SIMULATION_FINISH))

    def execute_command(self, command: str):
        self.sock.sendall(struct.pack("ii", MessageType.C_EXECUTE_COMMAND, np.int32(len(command))))
        self.sock.sendall(command.encode())  # https://www.delftstack.com/howto/python/python-socket-send-string/

    def set_timeout(self, new_timeout: int):
        self.sock.sendall(struct.pack("iI", MessageType.C_SET_TIMEOUT, np.uint32(new_timeout)))

    def set_speed(self, new_speed):
        self.sock.sendall(struct.pack("if", MessageType.C_SET_SPEED, np.float32(new_speed)))

    def race_finished(self):
        self.sock.sendall(struct.pack("i", MessageType.C_RACE_FINISHED))
        a = self._read_int32()
        return a

    def request_frame(self, W: int, H: int):
        self.sock.sendall(struct.pack("iii", MessageType.C_REQUEST_FRAME, np.int32(W), np.int32(H)))

    def unrequest_frame(self):
        self.sock.sendall(struct.pack("i", MessageType.C_UNREQUEST_FRAME))

    def get_frame(self, width: int, height: int) -> np.ndarray:
        frame_data = self.sock.recv(width * height * 4, socket.MSG_WAITALL)
        return np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 4))

    def toggle_interface(self, new_val: bool):
        self.sock.sendall(struct.pack("ii", MessageType.C_TOGGLE_INTERFACE, np.int32(new_val)))

    def set_on_step_period(self, period: int):
        self.sock.sendall(struct.pack("ii", MessageType.C_SET_ON_STEP_PERIOD, np.int32(period)))

    def is_in_menus(self):
        self.sock.sendall(struct.pack("i", MessageType.C_IS_IN_MENUS))
        return self._read_int32() > 0

    def get_inputs(self):
        self.sock.sendall(struct.pack("i", MessageType.C_GET_INPUTS))
        string_length = self._read_int32()
        return self.sock.recv(string_length, socket.MSG_WAITALL).decode("utf-8")

    def _respond_to_call(self, response_type):
        self.sock.sendall(struct.pack("i", np.int32(response_type)))

    def _read_int32(self) -> int:
        """
        Reads 4 bytes (32 bit) from tcp stream (MSG.WAITALL == waits until all 4 bytes are received before returning.)
        """
        bytes_from_tcp_stream = self.sock.recv(4, socket.MSG_WAITALL)
        (integer, ) = struct.unpack("i", bytes_from_tcp_stream)
        return integer
    
    def on_connect_event(self, user_profile : int = 1, map_to_load : str = "ESL-Hockolicious.Challenge.Gbx") -> None:
        """Executes command for game-initialization. Basically prompts the game to directly enter into the loaded map."""
        self.execute_command("toggle_console")
        self.execute_command(f"set autologin {user_profile}")
        self.execute_command("set unfocused_fps_limit false")
        self.execute_command("set disable_forced_camera true")
        self.execute_command("set autorewind false")
        self.execute_command("set auto_reload_plugins true")
        self.execute_command(f"map {map_to_load}")

import subprocess
import socket
import struct
import signal
from sys import platform

import subprocess
import socket
import time

def get_wsl_host_ip():
    """Extracts the Windows Host IP from the WSL routing table."""
    try:
        # Use 'ip route' to find the default gateway (the Windows side)
        cmd = "ip route show | grep default | awk '{print $3}'"
        host_ip = subprocess.check_output(cmd, shell=True).decode().strip()
        if not host_ip:
            return "127.0.0.1"
        return host_ip
    except Exception as e:
        print(f"Error finding host IP: {e}")
        return "127.0.0.1"

class TMInterfaceWSL(TMInterface):
    def __init__(self, port: int):
        super().__init__(port)
        self.host = get_wsl_host_ip()

    def register(self, timeout=None, retries=3):
        self.host = get_wsl_host_ip() # Refresh IP just in case
        
        for attempt in range(retries):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # Apply timeout logic here...
                
                self.logger.info(f"Connecting to {self.host}:{self.port} (Attempt {attempt+1})")
                self.sock.connect((self.host, self.port))
                self.registered = True
                self.logger.info("Successfully connected to Windows!")
                return
            except ConnectionRefusedError:
                if attempt < retries - 1:
                    self.logger.warning("Refused. Retrying in 2s...")
                    time.sleep(2)
                else:
                    self.logger.error("Connection failed. 1) Check TMInterface 'Listen Address' is 0.0.0.0. 2) Check Windows Firewall.")
                    raise

