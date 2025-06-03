import numpy as np

class TMInterfaceCommands:
    """This class implements commands from https://donadigo.com/tminterface/commands as methods with given parameters.
    All Methods return strings that can be sent via tminterface.execute_command(...)."""

    @staticmethod
    def recover_inputs(filename : str) -> str:
        """Calls 'recover inputs'-command, which saves them to the give filename (also add .txt-fileextension to the filename)
        located in the scripts/ folder."""
        return f"recover_inputs {filename}"
    
    @staticmethod
    def teleport(coordinates : list[float] | np.ndarray) -> str:
        """Teleports the car to specified x,y,z coordinates."""
        return f"tp {coordinates[0]} {coordinates[1]} {coordinates[2]}"
    
    @staticmethod
    def key_action(action : str, key : str) -> str:
        """Presses or releases a key.

            action : ' press' for pressing key and 'rel' for releasing key.
         
            Valid keys are : 
                - up, down, left, right, <- these are used for steering
                - enter : respawns the car
                - delete: restarts the current race      

            Note that a press-command should be followed by a release-command; except for 'enter' or 'delete', which do not need release command.      
        """
    
        assert key in ["up", "down", "left", "right", "enter", "delete"], f"Got unexpected key to press : {key}"
        assert action in ["press", "rel"], f"Got unexpected action for key : {action}"
        return f"{action} {key}" 
    
        

    