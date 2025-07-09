from game_interaction.tminterface2 import MessageType, TMInterface
from game_interaction.game_instance_manager2 import GameInstanceManager
from game_interaction.ipc_fields import IPCFields
import numpy as np
import time
import logging, os

from multiprocessing import Queue
from queue import Empty
from tminterface.structs import CheckpointData, SimStateData

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
        REWIND_STATE = 6
        STEP = 7
        WAITFORSTEP = 8

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
        
        @staticmethod
        def rewind_state(command_id : int, state : SimStateData) -> dict[str, any]:
            """Sends command to call ifcae.prevent_simulation_finish """
            return {IPCFields.CMD_ID : command_id, IPCFields.CMD : TMIProcessWrapper.IPCCommands.REWIND_STATE, IPCFields.ARGS : state}
        
        @staticmethod
        def step(command_id : int, action : tuple[bool, bool, bool, bool]) -> dict[str, any]:
            """Sends command to call ifcae.prevent_simulation_finish """
            return {IPCFields.CMD_ID : command_id, IPCFields.CMD : TMIProcessWrapper.IPCCommands.STEP, IPCFields.ARGS : action}

        @staticmethod
        def waitforstep(command_id : int, max_wait_duraion : float) -> dict[str, any]:
            """Sends command to setp process wrapper into stepping mode """
            return {IPCFields.CMD_ID : command_id, IPCFields.CMD : TMIProcessWrapper.IPCCommands.WAITFORSTEP, IPCFields.ARGS : max_wait_duraion}
        
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

        
        self._req_in_progress : bool = False
        """If True, request was sent to the tm-server but no image received yet."""

        self.sim_step_count = 0

        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("TMIProcessWrapper initialized")

        self.__run_sync_loop = True
        self.__start_cmd_id = -1

        self.automatic_prevent_sim_finish = automatic_prevent_sim_finish
        self.map = track
        self.logdir = "logs"


        self.waitforstep : bool = False
        self.max_waiting_duration : float = 1


        self.n_steps = 0
        """Tracks number of steps to track step-frequency"""
        self.step_time = time.time()
        """Tracks time to track step-frequency"""



        
    def __receive_frame(self):
        """Receives a frame from self.ifaca If self._req_img_cmd_id != -1, it sends the aquired frame and simstate via the response-queue"""
        frame = self.iface.get_frame(self.img_width, self.img_height)
        simstate = self.iface.get_simulation_state()

        self._req_in_progress = False
        
        response = {IPCFields.CMD_ID : self._req_img_cmd_id, IPCFields.STATUS : IPCFields.STATUS_OK, IPCFields.IMG : frame,IPCFields.SIMSTATE : simstate, IPCFields.SIMSTEP : self.sim_step_count}

        if not self._req_img_cmd_id == -1:
            self.response_queue.put_nowait(response)
            self._req_img_cmd_id = -1

        return response



    def send_action(self, action) -> dict:
        left, right, acc, brake = action
        self.iface.set_input_state(left, right, acc, brake)


    def stop_sync_loop(self) -> None:
        """Stops running syncloop(). May result in timeout-error."""
        self.__run_sync_loop = False

    def _reconfigure_logger(self, log_file : str):
        """This has to be called when executing because this is a sperate process from the main process, therefore needs own log-config."""
        if not os.path.exists(self.logdir):
            os.mkdir(self.logdir)

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(os.path.join(self.logdir, log_file), mode='w')]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.debug("Logger configured in subprocess.")

    def request_image(self):
        """Request an image with the specified image and width (specified in class initialization)        
        cmd_id : command-id for IPC; used internally only."""

        self.iface.request_frame(self.img_width, self.img_height)
        self._req_in_progress = True
        

    def check_command_queue(self) -> dict:
        """Checks command queue for ICP and handles command appropriately. Returns command."""
        # first get command
        assert self.command_queue is not None
        try:
            cmd = self.command_queue.get_nowait()
        except Empty:
            return None
        
        # now handle command
        cmd_id : int = cmd[IPCFields.CMD_ID]
        #self.logger.debug(f"Got command with command-id {cmd_id} of type {cmd['cmd']}") - TODO figure out if this causes huge log-files
        assert not cmd_id == -1, "Command id cannot be -1 as this is used as an internal error-code."
        command = cmd[IPCFields.CMD]
        if command == TMIProcessWrapper.IPCCommands.ACT:
            self.send_action(cmd[IPCFields.ARGS])
            self.response_queue.put_nowait({IPCFields.CMD_ID : cmd_id, IPCFields.STATUS : IPCFields.STATUS_OK})

        elif command == TMIProcessWrapper.IPCCommands.REQ_IMG:
            self._req_img_cmd_id = cmd_id
            self.request_image()
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
        elif command == TMIProcessWrapper.IPCCommands.REWIND_STATE:
            self.iface.rewind_to_state(cmd[IPCFields.ARGS])
            self.response_queue.put_nowait({IPCFields.CMD_ID : cmd_id, IPCFields.STATUS : IPCFields.STATUS_OK})


        elif command == TMIProcessWrapper.IPCCommands.STEP:
            assert self.waitforstepmode_on, "Cannot call step before waitfotsep has not been enabled."

            if not self.waitforstep:#<-- in case waitforstep disabled itself; this enables it again after it has been called.
                self.waitforstep_execution(cmd[IPCFields.ARGS], cmd_id = cmd_id)
                self.logger.warning("Self enabling self.waitforstep, as step method has been called again.")
            self.waitforstep = True 
                
            return cmd   
        elif command == TMIProcessWrapper.IPCCommands.WAITFORSTEP:
            self.step_time = time.time()
            self.waitforstep = True
            self.waitforstepmode_on = True #<-- Two variables because waitforstep can disable itself.
            self.max_waiting_duration = cmd[IPCFields.ARGS]
            self.logger.info(f"Enabling wait-for-step mode. Max-Waiting-Duration set to {self.max_waiting_duration}s")
            self.response_queue.put_nowait({IPCFields.CMD_ID : cmd_id, IPCFields.STATUS : IPCFields.STATUS_OK})
        else:
            self.response_queue.put_nowait({IPCFields.CMD_ID : cmd_id, IPCFields.STATUS : IPCFields.STATUS_ERROR, IPCFields.ERROR : "NoSuchCommand"})
        return cmd

    def waitforstep_execution(self, action, cmd_id : int):
        self.iface.rewind_to_current_state()
        self.waitforstep_step_cmd_id = cmd_id
        self.send_action(action)
        self.waitforstep_req_img_next_syncstep = True
        self.waitforstep_continue_sucessfully = True
        self.waitforstep_answer_expected = True
        self.ingame_time_passed = 0
        self.n_steps += 1
        if self.n_steps % 100 == 0 and self.n_steps > 0 :
            gt= time.time() - self.step_time
            self.logger.info(f"Executed {self.n_steps} in {gt}s. Actions per second : {self.n_steps / gt}s. And actions per ingame seconds game_seconds : {self.n_steps / (self.ingame_time_tracking / 1000)}")
            self.step_time = time.time()
            self.n_steps = 0
            self.ingame_time_tracking = 0


    def syncloop(self, logfile = "tmi_process.log"):
        self._reconfigure_logger(logfile)
        self.logger.info("Started syncloop.")

        
        self.waitforstep_req_img_next_syncstep = False
        frame_and_state = None
        self.waitforstep_step_cmd_id = -1
        self.waitforstep_answer_expected = False
        self.waitforstep_continue_sucessfully = False


        aps = 20 # actions per second.
        gametime_between_actions = 1 / aps * 1000 # convert actions per second into gametime 
        
        self.ingame_time_passed = 0
        global_ingame_time = 0
        self.ingame_time_tracking = 0

        total_msgs = 0
        unsucessful_waitings_for_step = 0
        disable_waitforstep_after_n_consecutive_timeouts = 10
        consecutive_timeouts = 0

        while self.__run_sync_loop:
            
            if self.waitforstep_req_img_next_syncstep:
                self.request_image()
                self.waitforstep_req_img_next_syncstep = False

            if self.waitforstep and not frame_and_state == None and self.waitforstep_answer_expected:
                frame_and_state[IPCFields.CMD_ID] = self.waitforstep_step_cmd_id
                self.response_queue.put_nowait(frame_and_state)
                frame_and_state = None
                self.waitforstep_step_cmd_id = -1
                self.waitforstep_answer_expected = False
                self.waitforstep_continue_sucessfully = False #only after waitforstep is officially done, this shoul be tracking again.

            broke_becauseof_req_img = False

            if self.waitforstep and not self._req_in_progress and not self.waitforstep_req_img_next_syncstep and not self.waitforstep_answer_expected and self.ingame_time_passed >= gametime_between_actions:

                waiting_duration = time.time()
                while  time.time() - waiting_duration < self.max_waiting_duration:
                    command = self.check_command_queue()
                    if not command == None and command[IPCFields.CMD] == TMIProcessWrapper.IPCCommands.STEP:
                        self.waitforstep_execution(command[IPCFields.ARGS], command[IPCFields.CMD_ID])
                        break
                        #if waiting_duration - time.time() < 1 / aps:
                        #    time.sleep(1 / aps - (waiting_duration - time.time()) - 0.00001)
                    if not command == None and command[IPCFields.CMD] == TMIProcessWrapper.IPCCommands.REQ_IMG:
                        broke_becauseof_req_img = True
                        break
                    time.sleep(0.0000001)
                


                if self.waitforstep and not broke_becauseof_req_img:
                    # self-disable mechanism for waitforstep
                    if self.waitforstep_continue_sucessfully:
                        consecutive_timeouts = 0
                    else:
                        consecutive_timeouts += 1
                    if consecutive_timeouts >= disable_waitforstep_after_n_consecutive_timeouts:
                        consecutive_timeouts = 0
                        self.waitforstep = False
                        self.logger.warning(f"Disabling waitforstep method after {disable_waitforstep_after_n_consecutive_timeouts} consecutive timeouts.")
                        self.send_action((False, False, False, False))

                unsucessful_waitings_for_step += (not self.waitforstep_continue_sucessfully and self.waitforstep)

                if total_msgs % 10000 == 0:
                    self.logger.info(f"Could not continue successfully. Timeout; This happend in {unsucessful_waitings_for_step}/{total_msgs} loop-iterations.")
                    total_msgs = 0 #<-- there is no need to set this to zero here but i am anxious it will overflow.
                    unsucessful_waitings_for_step = 0

            msgtype = self.iface._read_int32()
            total_msgs += 1
            
            # ============================================= READ INCOMING MESSAGES
            
            if msgtype == int(MessageType.SC_RUN_STEP_SYNC): # simulation step is complete

                _time = self.iface._read_int32() # _time in this case is the total simulation time (i think)

                if _time > 0:
                    increment = _time - global_ingame_time
                    self.ingame_time_tracking += increment
                    self.ingame_time_passed += increment
                    global_ingame_time = _time

                else:
                    global_ingame_time = 0
                    self.ingame_time_passed = 0
                    self.ingame_time_tracking = 0

                self.sim_step_count += 1

                # ============================ BEGIN ON RUN STEP ============================

                if not self.__start_cmd_id == -1:
                    self.response_queue.put_nowait({IPCFields.CMD_ID : self.__start_cmd_id, IPCFields.STATUS : IPCFields.STATUS_OK})
                    self.__start_cmd_id = -1
                    self.logger.info("Sending back command that abcdefg is ready.")

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
                frame_and_state = self.__receive_frame()
                self.iface._respond_to_call(msgtype)

            elif msgtype == int(MessageType.C_SHUTDOWN):

                self.iface.close()

            elif msgtype == int(MessageType.SC_ON_CONNECT_SYNC):
                self.logger.info("On connect event. Reuesting map.")
                self.iface.on_connect_event(map_to_load=self.map)
                self.iface._respond_to_call(msgtype)
            else:
                self.iface._respond_to_call(msgtype)

            if not self.waitforstep:
                self.check_command_queue()

        if self.launch_game:
            self.gim.close_game()
        

        


    