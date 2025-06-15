

class IPCFields:
    """
    This ENUM-Class specifies strings used as keys in dictionarys that are sent between processes. 
    When the TMIProcessWrapper is run, its possible to send commands and ceie responses in dicts via the initiated queues.
    """

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