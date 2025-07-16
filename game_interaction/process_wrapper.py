from game_interaction.tminterface2 import MessageType, TMInterface
from game_interaction.game_instance_manager2 import GameInstanceManager
from game_interaction.ipc_fields import IPCFields
from game_interaction.tminterface_commands import TMInterfaceCommands
import numpy as np
import time
import logging, os

from multiprocessing import Queue as MultiprocessingQueue
from queue import Empty, Queue 
from tminterface.structs import CheckpointData, SimStateData
from configs.config import EnvConfig

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
                 command_queue : MultiprocessingQueue, response_queue : MultiprocessingQueue, 
                 track : str, 
                 img_width : int, img_height : int,
                 config : EnvConfig):
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
        - config : Environment config; contains more configuration
        """
        self.launch_game : bool = launch_game
        self.gim = gim
        if launch_game:
            self.gim.launch_game()
            self.gim.register_iface(10)
        self.iface : TMInterface = gim.get_tminterface()
        self.command_queue : MultiprocessingQueue = command_queue
        self.response_queue : MultiprocessingQueue = response_queue

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

        self.automatic_prevent_sim_finish = config.automatic_prevent_sim_finish
        """If True, iface.prevent_sim_finish() is called automatically, once the current and target checkpoint are the same."""

        self.map = track
        self.logdir = "logs"


        self.waitforstep : bool = False

        self.ui_disabled = False

        self.n_steps = 0
        """Tracks number of steps to track step-frequency"""
        self.step_time = time.time()
        """Tracks time to track step-frequency"""

        self.aps = config.actions_per_second
        self.gametime_between_actions = 1 / self.aps * 1000 # in-game-seconds between each action
        self.disable_waitforstep_after_n_consecutive_timeouts = config.disable_waitforstep_after_n_consecutive_timeouts
        self.use_rewind  = config.use_rewind
        self.camera_id = config.camera_id

        self.waitforstep_step_cmd_id = -1
        self.waitforstep_answer_expected = False
        self.waitforstepmode_on = False
        self.max_waiting_period_for_step_command = 0.5

        self.time_since_last_command_check = 0
        """This is the last time the command-queue was cheked"""
        self.max_time_between_command_check = 30
        """Maximum time [in seconds] which can lay between two command-queue checks"""

        self.last_received_command_id = -1
        """Command-id of the command that was last received."""


        self.step_commands = Queue()
        self.unanswered_commands : dict[int, float] = {}
        """Holds all unanswered commands. Keys : cmd_id, value : time of arrival. Is used to send error codes, if command has not been answered in certain time."""

        self.time_since_last_unanswered_handling = time.time()
        """Time since all last unanswered commands were handled"""

        self.cmd_count, self.avg_cmd_exectime = 0, 0.0
        """These two log average execution times of all commands."""
        
    def __receive_frame(self) -> dict:
        """Receives a frame from self.ifaca If self._req_img_cmd_id != -1, it sends the aquired frame and simstate via the response-queue.
        If receive_frame was triggered by req_img-command, it also puts the answer into the return queue, if it was triggered by waitforstep, it returns the answer."""
        frame = self.iface.get_frame(self.img_width, self.img_height)
        simstate = self.iface.get_simulation_state()

        self._req_in_progress = False
        
        response = {IPCFields.CMD_ID : self._req_img_cmd_id, IPCFields.STATUS : IPCFields.STATUS_OK, IPCFields.IMG : frame, IPCFields.SIMSTATE : simstate, IPCFields.SIMSTEP : self.sim_step_count}

        if not self._req_img_cmd_id == -1: #<- tis is only set by req-img-command
            self.answer_command(response)
            self._req_img_cmd_id = -1
        else: #<- if it's not set, then Process wrapper is in waitforstep-mode
            return response



    def send_action(self, action : tuple[bool, bool, bool, bool]) -> dict:
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
        if self.use_rewind:
            self.iface.rewind_to_current_state()
        self.iface.request_frame(self.img_width, self.img_height)
        self._req_in_progress = True

    def answer_command(self, response : dict):
        """Answers a command. Delets it from unanswered commands and sends puts response into queue."""
        if not response[IPCFields.CMD_ID] in self.unanswered_commands:
            self.logger.error(f"Wanted to answer command with id {response[IPCFields.CMD_ID]} that was not in unanswered commands; has to have been removed by unanswered-handler.")
            return 
        exce_time = self.unanswered_commands[response[IPCFields.CMD_ID]]
        del self.unanswered_commands[response[IPCFields.CMD_ID]]
        self.cmd_count += 1
        self.avg_cmd_exectime += ((time.time() - exce_time) - self.avg_cmd_exectime) / self.cmd_count
        if self.cmd_count % 10000 == 0:
            self.logger.info(f"Average execution time of IPC-Commands {self.avg_cmd_exectime}s.")
        self.response_queue.put_nowait(response)

    def handle_unanswered_commands(self, max_unanswered_time : float = 20):
        """Removes all commands that are unanswered for more than"""
        for cmd_id in self.unanswered_commands:
            if time.time() - self.unanswered_commands[cmd_id] > max_unanswered_time:
                self.logger.error(f"Got a command that was unanswered for more than {max_unanswered_time} seconds. Command id: {cmd_id}")
                self.response_queue.put_nowait({IPCFields.CMD_ID : cmd_id, IPCFields.STATUS : IPCFields.STATUS_ERROR, IPCFields.ERROR : "Could not answer command in given timeframe."})
                del self.unanswered_commands[cmd_id]
        

    def check_command_queue(self) -> dict:
        """Checks command queue for ICP and handles command appropriately. Returns command."""
        # first get command
        assert self.command_queue is not None
        try:
            cmd = self.command_queue.get_nowait()
        except Empty:
            self.time_since_last_command_check = time.time()
            return None
        
        # now handle command
        cmd_id : int = cmd[IPCFields.CMD_ID]
        assert not cmd_id == -1, "Command id cannot be -1 as this is used as an internal error-code."

        if cmd_id == self.last_received_command_id:
            # just skip commands with doubled command-ids (commands may be sent more than once by other processes; due to error-handling e.g.)
            self.logger.warning(f"Got command with id {cmd_id} more than once. Ignoring.")
            return None

        command = cmd[IPCFields.CMD]
        self.unanswered_commands[cmd_id] = time.time()

        if command == TMIProcessWrapper.IPCCommands.ACT:
            self.send_action(cmd[IPCFields.ARGS])
            self.answer_command({IPCFields.CMD_ID : cmd_id, IPCFields.STATUS : IPCFields.STATUS_OK})

        elif command == TMIProcessWrapper.IPCCommands.REQ_IMG:
            self._req_img_cmd_id = cmd_id
            self.request_image()
        elif command == TMIProcessWrapper.IPCCommands.END_SYNCLOOP: 
            self.stop_sync_loop()
            self.answer_command({IPCFields.CMD_ID : cmd_id, IPCFields.STATUS : IPCFields.STATUS_OK})
        elif command == TMIProcessWrapper.IPCCommands.EXECUTE_COMMAND:
            self.iface.execute_command(cmd[IPCFields.ARGS])
            self.answer_command({IPCFields.CMD_ID : cmd_id, IPCFields.STATUS : IPCFields.STATUS_OK})
        elif command == TMIProcessWrapper.IPCCommands.SIMULATION_STARTED:
            self.__start_cmd_id = cmd[IPCFields.CMD_ID]
        elif command == TMIProcessWrapper.IPCCommands.REVENT_SIM_FINISH:
            self.iface.prevent_simulation_finish()
            self.answer_command({IPCFields.CMD_ID : cmd_id, IPCFields.STATUS : IPCFields.STATUS_OK})
        elif command == TMIProcessWrapper.IPCCommands.REWIND_STATE:
            self.iface.rewind_to_state(cmd[IPCFields.ARGS])
            self.answer_command({IPCFields.CMD_ID : cmd_id, IPCFields.STATUS : IPCFields.STATUS_OK})

        elif command == TMIProcessWrapper.IPCCommands.STEP:
            assert self.waitforstepmode_on, "Cannot call step before waitfotsep has not been enabled."
            if not self.waitforstep:
                self.logger.warning("Self enabling self.waitforstep, as step method has been called again.")

            self.waitforstep = True
            self.step_commands.put_nowait(cmd)
        elif command == TMIProcessWrapper.IPCCommands.WAITFORSTEP:
            self.step_time = time.time()
            self.waitforstep = True
            self.waitforstepmode_on = True
            self.expect_next_step_command = True
            self.max_waiting_period_for_step_command = cmd[IPCFields.ARGS]
            self.answer_command({IPCFields.CMD_ID : cmd_id, IPCFields.STATUS : IPCFields.STATUS_OK})
        else:
            self.answer_command({IPCFields.CMD_ID : cmd_id, IPCFields.STATUS : IPCFields.STATUS_ERROR, IPCFields.ERROR : "NoSuchCommand"})
            
        self.last_received_command_id = cmd_id
        self.time_since_last_command_check = time.time()
        return cmd

    def waitforstep_execution(self, action, cmd_id : int):
        self.waitforstep_step_cmd_id = cmd_id
        self.send_action(action)
        self.waitforstep_req_img_next_syncstep = True
        self.expect_next_step_command = False
        self.ingame_time_passed = 0
        self.n_steps += 1
        if self.n_steps % 100 == 0 and self.n_steps > 0:
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

        self.expect_next_step_command = False

        self.ingame_time_passed = 0
        global_ingame_time = 0
        self.ingame_time_tracking = 0

        total_msgs = 0
        unsucessful_waitings_for_step = 0
        continuous_step_misses = 0
        sepcommand_count, avg_waitingtime_for_command = 0, 0.0  # alpha, avg_time = 0.01, 0.0 #for moving average.
        """This tracks the average waiting time for step-commands"""

    
        self.time_since_last_unanswered_handling = time.time()

        while self.__run_sync_loop:
            
            # -------- methods for waitforstep

            if self.waitforstep:
                assert self.waitforstep_req_img_next_syncstep or self.waitforstep_answer_expected or self.expect_next_step_command

            if self.waitforstep_req_img_next_syncstep:
                self.request_image()
                self.waitforstep_req_img_next_syncstep = False
                self.waitforstep_answer_expected = True

            if self.waitforstep_answer_expected and not frame_and_state == None:
                frame_and_state[IPCFields.CMD_ID] = self.waitforstep_step_cmd_id
                self.answer_command(frame_and_state)
                frame_and_state = None
                self.waitforstep_step_cmd_id = -1
                self.waitforstep_answer_expected = False
                self.expect_next_step_command = True

            if self.waitforstep and self.expect_next_step_command and not self._req_in_progress and self.ingame_time_passed >= self.gametime_between_actions: #'not self._req_in_progress' because if req-img command is currently handled, this should not block
                waiting_begin = time.time()
                waiting_time = time.time() - waiting_begin
                waiting_period = 1.0
                break_due_to_req_img = False
                while waiting_time < waiting_period:
                    try:
                        command = self.step_commands.get_nowait()
                        self.waitforstep_execution(command[IPCFields.ARGS], command[IPCFields.CMD_ID])
                        continuous_step_misses = 0
                        
                        sepcommand_count += 1
                        avg_waitingtime_for_command += (waiting_time - avg_waitingtime_for_command) / sepcommand_count # moving average?: alpha * waiting_time + (1-alpha) * avg_time
                        if sepcommand_count % 10000 == 0:
                            self.logger.info(f"Average waiting time for step-command : {avg_waitingtime_for_command}s")
                        break
                    except Empty:
                        pass

                    _command = self.check_command_queue()
                    if not _command is None and _command[IPCFields.CMD] == TMIProcessWrapper.IPCCommands.REQ_IMG:
                        # if _command is req_img, then it has to be handled in the communication loop; this is why it has to be broken here.
                        # TODO : originally, this was only for breaking to handle image, but maybe it's not so bad to generally update the simulation if a command happened.
                        break_due_to_req_img = True
                        break
                        


                    waiting_time = time.time() - waiting_begin

                continuous_step_misses += (not break_due_to_req_img)


                # self-disable mechanism for waitforstep
                if continuous_step_misses >= self.disable_waitforstep_after_n_consecutive_timeouts:
                    continuous_step_misses = 0
                    self.ingame_time_passed = 0
                    self.n_steps = 0
                    self.waitforstep = False
                    self.logger.warning(f"Got no step command for more than {self.disable_waitforstep_after_n_consecutive_timeouts} Tries.")
                    self.send_action((False, False, False, False))



            if self.waitforstep and total_msgs % 10000 == 0 and unsucessful_waitings_for_step > 0:
                self.logger.info(f"Had no step-command in internal queue. This happend in {unsucessful_waitings_for_step}/{10000} loop-iterations.")
                total_msgs = 0 #<-- there is no need to set this to zero here but i am anxious it will overflow.
                unsucessful_waitings_for_step = 0

            #--------- end mehtods waitforstep

            msgtype = self.iface._read_int32()
            total_msgs += 1
            
            # ============================================= READ INCOMING MESSAGES
            
            if msgtype == int(MessageType.SC_RUN_STEP_SYNC): # simulation step is complete

                _time = self.iface._read_int32() # _time in this case is the total simulation time since last respawn
                
                if _time > 0: # if _time < 0, countdown is running
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

                self.iface.execute_command(TMInterfaceCommands.set_camera(self.camera_id)) # TODO : is this necessary on every step?

                if not self.ui_disabled:
                    self.iface.toggle_interface(False)
                    self.ui_disabled = True

                if not self.__start_cmd_id == -1:
                    self.answer_command({IPCFields.CMD_ID : self.__start_cmd_id, IPCFields.STATUS : IPCFields.STATUS_OK})
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

            
            self.check_command_queue()
            if time.time() - self.time_since_last_unanswered_handling > self.max_time_between_command_check:
                self.handle_unanswered_commands()

        if self.launch_game:
            self.gim.close_game()