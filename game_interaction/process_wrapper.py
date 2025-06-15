from game_interaction.tminterface2 import MessageType, TMInterface
from game_interaction.game_instance_manager2 import GameInstanceManager
from game_interaction.ipc_fields import IPCFields
import numpy as np
import time
import logging, os

from multiprocessing import Queue
from queue import Empty

class TMIProcessWrapper:

    """
    TMIProcessWrapper encompasses the interaction between python (client) and trackmania instance (server) as a seperately executable process. 
    After initialization, the syncloop method can be started and runs the communication to not cause timeouts or blockings in other processes.

    In order to accomplish this, a command and response queue are set-up for Inter-Process-Communication (IPC).
    """

    class IPCCommands:
        """Helper class to get commands in order to have one place where commands are generated."""

        ACT = 0
        REQ_IMG = 1
        END_SYNCLOOP = 2
        EXECUTE_COMMAND = 3
        SIMULATION_STARTED = 4
        REVENT_SIM_FINISH = 5

        @staticmethod
        def get_act_command(command_id : int, action : tuple[bool, bool, bool, bool]) -> dict[str, any]:
            """Creates command for IPC communication for request_image()"""
            return {IPCFields.CMD_ID : command_id, IPCFields.CMD : TMIProcessWrapper.IPCCommands.ACT, IPCFields.ARGS : action}
            
        @staticmethod
        def get_req_img_command(command_id : int, continuous : bool = False) -> dict[str, any]:
            """Creates command for IPC communication for request_image()"""
            return {IPCFields.CMD_ID : command_id, IPCFields.CMD : TMIProcessWrapper.IPCCommands.REQ_IMG, IPCFields.ARGS : continuous}
        
        @staticmethod
        def get_end_syncloop_command(command_id : int) -> dict[str, any]:
            """Returns command to end syncloop execution."""
            return {IPCFields.CMD_ID : command_id, IPCFields.CMD : TMIProcessWrapper.IPCCommands.END_SYNCLOOP, IPCFields.ARGS : None}
        
        @staticmethod
        def get_cmd_command(command_id : int, command : str) -> dict[str, any]:
            """Returns a command (for TMIProcessWrapper) that tells the underlying TMInterface to send a command to the game-instance."""
            return {IPCFields.CMD_ID : command_id, IPCFields.CMD : TMIProcessWrapper.IPCCommands.EXECUTE_COMMAND, IPCFields.ARGS : command}
        
        @staticmethod
        def get_startsignal(command_id : int) -> dict[str, any]:
            """Returns a command which only sends a response, once the server is sending `SC_SYNC' commands; i.e. track has started."""
            return {IPCFields.CMD_ID : command_id, IPCFields.CMD : TMIProcessWrapper.IPCCommands.SIMULATION_STARTED}
        
        @staticmethod
        def prevent_simulation_finish(command_id : int) -> dict[str, any]:
            """Sends command to call ifcae.prevent_simulation_finish """
            return {IPCFields.CMD_ID : command_id, IPCFields.CMD : TMIProcessWrapper.IPCCommands.REVENT_SIM_FINISH}


    def __init__(self, gim : GameInstanceManager, launch_game : bool, 
                 command_queue : Queue, response_queue : Queue, 
                 track : str, 
                 img_width : int, img_height : int,
                 automatic_prevent_sim_finish : bool = True):
        """
        Parameters
        ---------
        - gim               : INtance of GameInstanceManager, from which TMInterface is aquired using gim.get_tminterface.
        - launch_game       : If True, gim.launc_game() is called and gim.register_iface(), also closes the game after end-syncloop-command is sent.
        - command_queue     : Queue from which commands are queried. Use IPCCommands for correct format.
        - response_queue    : Queue used for sending responses to commands.
        - track             : Specifies which track to load once instance is connected to game
        - img_width         : Image width of images queried from game
        - img_height        : Image height of images queried from game
        - automatic_prevent_sim_finish : If True, iface.prevent_sim_finish() is called automatically, once the current and target checkpoint are the same.
        """
        self.launch_game : bool = launch_game
        self.gim = gim
        if launch_game:
            self.gim.launch_game()
            self.gim.register_iface(10)
        self.iface : TMInterface = gim.get_tminterface()
        self.command_queue : Queue = command_queue
        self.response_queue : Queue = response_queue

        self._act_cmd_id = -1
        self._req_img_cmd_id = -1

        self.img_width = img_width
        self.img_height = img_height

        
        self._req_img : bool = False
        self.__continuous_image_request : bool = False
        self._req_in_progress : bool = False
        """If True, request was sent to the tm-server but no image received yet."""

        self.sim_step_count = 0

        self._send_action = False
        self.action : tuple[bool, bool, bool, bool] = None
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("TMIProcessWrapper initialized")

        self.__run_sync_loop = True
        self.__start_cmd_id = -1

        self.automatic_prevent_sim_finish = automatic_prevent_sim_finish
        self.map = track

    def request_image(self, continuously : bool = False, cmd_id : int = -1):
        """Request an image with the specified image and width (specified in class initialization)
        
        If continuously is True, request_image does not have to be called again and again, but rather always requests images.
        
        cmd_id : command-id for IPC; used internally only."""
        self.__continuous_image_request = continuously
        self._req_img = True
        self._req_img_cmd_id = cmd_id
        self.__img_req_step_count = self.sim_step_count
        
    def __receive_frame(self):
        self.logger.debug("Receiving frame.")
        frame = self.iface.get_frame(self.img_width, self.img_height)


        self._req_in_progress = False
        self.logger.warning(f"Image was requested in stepcount {self.__img_req_step_count} and was received in {self.sim_step_count}")

        #if self.__continuous_image_request is True, self._req_img just stays True, if its false, its reset to false and self.request_image() has to be called again.
        self._req_img = self.__continuous_image_request

        ssD = self.iface.get_simulation_state()

        if not self._req_img_cmd_id == -1:
            self.response_queue.put_nowait({IPCFields.CMD_ID : self._req_img_cmd_id, IPCFields.STATUS : IPCFields.STATUS_OK, 
                                            IPCFields.IMG : frame,
                                            IPCFields.SIMSTATE : ssD,
                                            IPCFields.SIMSTEP : self.sim_step_count})

    def __request_frame(self):
        self.logger.debug("Requesting frame from game instance.")
        self.iface.request_frame(self.img_width, self.img_height)
        self._req_in_progress = True

    def act(self, action : tuple[bool, bool, bool, bool], cmd_id : int = -1) -> int:
        """Sends action to trackmania game. action : to be sen @self.iface.set_input_state for more info. 
        cmd_id : used internally for IPC."""
        self.logger.debug(f"act() method is called with action: {action}")
        self.action = action
        self._send_action = True
        self._anticipated_simulation_step_of_execution = self.sim_step_count + 1
        self._act_cmd_id = cmd_id
        return self._anticipated_simulation_step_of_execution
    
    def __send_action(self) -> None:
        self.logger.debug(f"Sending Action {self.action}")

        left, right, acc, brake = self.action
        if not self._anticipated_simulation_step_of_execution == self.sim_step_count:
            self.logger.warning(f"Anticipated to execute action on simulation step {self._anticipated_simulation_step_of_execution}, but actual simulation step was {self.sim_step_count}")
        self.iface.set_input_state(left, right, acc, brake)
        self._send_action = False

        if not self._act_cmd_id == -1:
            self.response_queue.put_nowait({IPCFields.CMD_ID : self._act_cmd_id, IPCFields.STATUS : IPCFields.STATUS_OK})

    def stop_sync_loop(self) -> None:
        """Stops running syncloop(). May result in timeout-error."""
        self.__run_sync_loop = False

    def _reconfigure_logger(self, log_file : str):
        """This has to be called when executing because this is a sperate process from the main process, therefore needs own log-config."""

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file, mode='w')]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.debug("Logger configured in subprocess.")

    def check_command_queue(self) -> None:
        """Checks command queue for ICP and handles command appropriately."""
        # first get command
        assert self.command_queue is not None
        try:
            cmd = self.command_queue.get_nowait()
        except Empty:
            return
        
        # now handle command
        cmd_id : int = cmd[IPCFields.CMD_ID]
        self.logger.debug(f"Got command with command-id {cmd_id} of type {cmd['cmd']}")
        assert not cmd_id == -1, "Command id cannot be -1 as this is used as an internal error-code."
        command = cmd[IPCFields.CMD]
        if command == TMIProcessWrapper.IPCCommands.ACT:
            self.act(cmd[IPCFields.ARGS], cmd_id)
        elif command == TMIProcessWrapper.IPCCommands.REQ_IMG:
            self.request_image(cmd[IPCFields.ARGS], cmd_id)
        elif command == TMIProcessWrapper.IPCCommands.END_SYNCLOOP: 
            self.stop_sync_loop()
            self.response_queue.put_nowait({IPCFields.CMD_ID : cmd_id, IPCFields.STATUS : IPCFields.STATUS_OK})
        elif command == TMIProcessWrapper.IPCCommands.EXECUTE_COMMAND:
            self.iface.execute_command(cmd[IPCFields.ARGS])
            self.response_queue.put_nowait({IPCFields.CMD_ID : cmd_id, IPCFields.STATUS : IPCFields.STATUS_OK})
        elif command == TMIProcessWrapper.IPCCommands.SIMULATION_STARTED:
            self.__start_cmd_id = cmd[IPCFields.CMD_ID]
        elif command == TMIProcessWrapper.IPCCommands.REVENT_SIM_FINISH:
            self.iface.prevent_simulation_finish()
            self.response_queue.put_nowait({IPCFields.CMD_ID : cmd_id, IPCFields.STATUS : IPCFields.STATUS_OK})
        else:
            self.response_queue.put_nowait({IPCFields.CMD_ID : cmd_id, IPCFields.STATUS : IPCFields.STATUS_ERROR, IPCFields.ERROR : "NoSuchCommand"})


    def syncloop(self, logfilepath = "logs/tmi_process.log"):
        self._reconfigure_logger(logfilepath)
        self.logger.info("Started syncloop.")

        while self.__run_sync_loop:
            if self.sim_step_count % 500 == 0:
                self.logger.debug(f"Sim-Step-Count at {self.sim_step_count}")

            msgtype = self.iface._read_int32()
            
            # ============================================= READ INCOMING MESSAGES
            
            
            if msgtype == int(MessageType.SC_RUN_STEP_SYNC): # simulation step is complete

                _time = self.iface._read_int32() # _time in this case is the total simulation time (i think)
                self.sim_step_count += 1

                # ============================ BEGIN ON RUN STEP ============================

                if not self.__start_cmd_id == -1:
                    self.response_queue.put_nowait({IPCFields.CMD_ID : self.__start_cmd_id, IPCFields.STATUS : IPCFields.STATUS_OK})
                    self.__start_cmd_id = -1
                    self.logger.info("Sending back command that abcdefg is ready.")

                if self._req_img and not self._req_in_progress:
                    self.__request_frame()

                if self._send_action:
                    self.__send_action()

                # ============================ END ON RUN STEP ============================
                self.iface._respond_to_call(msgtype)


            elif msgtype == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):

                current = self.iface._read_int32()
                target = self.iface._read_int32()

                # ============================ BEGIN ON CP COUNT ============================
                if current == target and self.automatic_prevent_sim_finish:  # Finished the race !!
                    self.logger.info("Called automatic self.iface.prevent-simulation-finish.")
                    self.iface.prevent_simulation_finish()
                # ============================ END ON CP COUNT ============================
                self.iface._respond_to_call(msgtype)
            elif msgtype == int(MessageType.SC_LAP_COUNT_CHANGED_SYNC):

                self.iface._respond_to_call(msgtype)

            elif msgtype == int(MessageType.SC_REQUESTED_FRAME_SYNC):
                self.__receive_frame()
                self.iface._respond_to_call(msgtype)

            elif msgtype == int(MessageType.C_SHUTDOWN):

                self.iface.close()

            elif msgtype == int(MessageType.SC_ON_CONNECT_SYNC):
                self.logger.info("On connect event. Reuesting map.")
                self.iface.on_connect_event(map_to_load=self.map)
                self.iface._respond_to_call(msgtype)
            else:
                self.iface._respond_to_call(msgtype)


            self.check_command_queue()

        if self.launch_game:
            self.gim.close_game()
        

        


    