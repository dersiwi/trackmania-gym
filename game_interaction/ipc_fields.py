from tminterface.structs import SimStateData


class IPCFields:
    """
    This ENUM-Class specifies strings used as keys in dictionarys that are sent between processes. 
    When the TMIProcessWrapper is run, its possible to send commands and ceie responses in dicts via the initiated queues.
    """

    ACTION = "action"
    
    IMG = "img"
    """Contains image aquired from game"""

    SIMSTATE = "sim_state"
    """Contains SimStateData object"""

    ARGS = "args"
    """When sending commands, this field contains possible arguments. May not be included in every command."""


    CMD_ID = "cmd_id"
    """Command id is used both in command and response. This can be used to check if the current response matches the sent command id."""
    
    CMD = "cmd"
    """Specifies the command that is sent to the process"""

    STATUS = "status"
    """Specifies the status of the response. 0 is ok, -1 is error."""
    STATUS_OK = 0
    STATUS_ERROR = -1

    ERROR = "error"
    """If STATUS == STATUS_ERROR, this field is included in the response, which contains the error message."""

    SIMSTEP = "sim_step"
    """This is sent in the response when requesting images and simstates, along with IMG and SIMSTATE. This is the current
    simulation step of the trackmania instance."""

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
        SET_ACTIONMODE = 9

        @staticmethod
        def get_act_command(command_id : int, action : tuple[bool, bool, bool, bool]) -> dict[str, any]:
            """Creates command for IPC communication for request_image()"""
            return {IPCFields.CMD_ID : command_id, IPCFields.CMD : IPCCommands.ACT, IPCFields.ARGS : action}
            
        @staticmethod
        def get_req_img_command(command_id : int, continuous : bool = False) -> dict[str, any]:
            """Creates command for IPC communication for request_image()"""
            return {IPCFields.CMD_ID : command_id, IPCFields.CMD : IPCCommands.REQ_IMG, IPCFields.ARGS : continuous}
        
        @staticmethod
        def get_end_syncloop_command(command_id : int) -> dict[str, any]:
            """Returns command to end syncloop execution."""
            return {IPCFields.CMD_ID : command_id, IPCFields.CMD : IPCCommands.END_SYNCLOOP, IPCFields.ARGS : None}
        
        @staticmethod
        def get_cmd_command(command_id : int, command : str) -> dict[str, any]:
            """Returns a command (for TMIProcessWrapper) that tells the underlying TMInterface to send a command to the game-instance."""
            return {IPCFields.CMD_ID : command_id, IPCFields.CMD : IPCCommands.EXECUTE_COMMAND, IPCFields.ARGS : command}
        
        @staticmethod
        def get_startsignal(command_id : int) -> dict[str, any]:
            """Returns a command which only sends a response, once the server is sending `SC_SYNC' commands; i.e. track has started."""
            return {IPCFields.CMD_ID : command_id, IPCFields.CMD : IPCCommands.SIMULATION_STARTED}
        
        @staticmethod
        def prevent_simulation_finish(command_id : int) -> dict[str, any]:
            """Sends command to call ifcae.prevent_simulation_finish """
            return {IPCFields.CMD_ID : command_id, IPCFields.CMD : IPCCommands.REVENT_SIM_FINISH}
        
        @staticmethod
        def rewind_state(command_id : int, state : SimStateData) -> dict[str, any]:
            """Sends command to rewind the state to the given simstate """
            return {IPCFields.CMD_ID : command_id, IPCFields.CMD : IPCCommands.REWIND_STATE, IPCFields.ARGS : state}
        
        @staticmethod
        def step(command_id : int, action : tuple[bool, bool, bool, bool]) -> dict[str, any]:
            """Sends a complete MDP step-command. Given action will be sent to agent, returns images and gamestates """
            return {IPCFields.CMD_ID : command_id, IPCFields.CMD : IPCCommands.STEP, IPCFields.ARGS : action}

        @staticmethod
        def waitforstep(command_id : int, max_wait_duraion : float) -> dict[str, any]:
            """Sends command to setp process wrapper into stepping mode """
            return {IPCFields.CMD_ID : command_id, IPCFields.CMD : IPCCommands.WAITFORSTEP, IPCFields.ARGS : max_wait_duraion}
        
        @staticmethod
        def set_actionmode(command_id : int, mode : int) -> dict[str, any]:
            """Sest the process-wrapper to expect eithert continuous or discrete actions. Default is discrete!
            @see from trackmania_env.utils.actionmap import ActionMode the modes you can choose from."""
            return {IPCFields.CMD_ID : command_id, IPCFields.CMD : IPCCommands.SET_ACTIONMODE, IPCFields.ARGS : mode}
