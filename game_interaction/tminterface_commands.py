

class TMInterfaceCommands:
    """This class implements commands from https://donadigo.com/tminterface/commands as methods with given parameters.
    All Methods return strings that can be sent via tminterface.execute_command(...)."""

    @staticmethod
    def recover_inputs(filename : str) -> str:
        """Calls 'recover inputs'-command, which saves them to the give filename (also add .txt-fileextension to the filename)
        located in the scripts/ folder."""
        return f"recover_inputs {filename}"