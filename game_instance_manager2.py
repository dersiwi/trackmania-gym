
import subprocess
import os
import time 
import psutil
import win32process
import win32gui 
from pathlib import Path

from tminterface2 import TMInterface

class GameInstanceManager:

    def __init__(self, TMLoader_path : str, TMLoader_profile_name : str, path_to_plugin : str, is_linux : bool = False):
        self.TMLoader_path : str = TMLoader_path
        self.TMLoader_profile_name = TMLoader_profile_name
        self.path_to_plugin = path_to_plugin
        self.is_linux = is_linux
        self.tmi_port = 8775
        self.tm_process_id = None

        self.tminterface = TMInterface(self.tmi_port)

    def register_iface(self, timeout_in_s : int = 10) -> None:
        """Calls self.tminterface.register(timeout). Is blocking as long as tminterface is not registered or ConnectionRefusedError."""
        if not self.tminterface.registered:
            while True:
                try:
                    self.tminterface.register(10)
                    break
                except ConnectionRefusedError as e:
                    print(e)

    def get_tminterface(self) -> TMInterface:
        return self.tminterface


    def __get_launch_string(self) -> str:
        launch_string = (
                'powershell -executionPolicy bypass -command "& {'
                f" $process = start-process -FilePath '{self.TMLoader_path}'"
                " -PassThru -ArgumentList "
                f'\'run TmForever "{self.TMLoader_profile_name}" /configstring=\\"set custom_port {self.tmi_port}\\"\';'
                ' echo exit $process.id}"'
            )
        return launch_string
    
    def _get_tm_window_id(self) -> None:
        assert self.tm_process_id is not None

        def get_hwnds_for_pid(pid):
            def callback(hwnd, hwnds):
                _, found_pid = win32process.GetWindowThreadProcessId(hwnd)

                if found_pid == pid:
                    hwnds.append(hwnd)
                return True

            hwnds = []
            win32gui.EnumWindows(callback, hwnds)
            return hwnds

        begin = time.time()
        while True:
            for hwnd in get_hwnds_for_pid(self.tm_process_id):
                if win32gui.GetWindowText(hwnd).startswith("Track"):
                    self.tm_window_id = hwnd
                    print(f"Found Trackmania process id: {self.tm_window_id=}")
                    return
            if time.time() - begin > 10:
                raise WindowsError("Could not find window within 10s.")

    def _run_launchstring_and_set_process_id(self) -> None:
        """
        This method gets the launch-string via __get_launch_string() and creates a process. 
        After launching the process it searches processes until it finds one that starts with 'TMForever'.
        Once this has been found, self.tmi_proces_id is set and the method returns.

        If process is not found within 10s after launching, ProcessLookupError is thrown.
        """
        assert os.path.exists(self.path_to_plugin), f"Python_Link.as was not found at '{self.path_to_plugin}'."
        tmi_process_id = int(subprocess.check_output(self.__get_launch_string()).decode().split("\r\n")[1])
        start_time = time.time()
        while self.tm_process_id is None:
            tm_processes = list(
                filter(
                    lambda s: s.startswith("TmForever"),
                    subprocess.check_output(["powershell", "-Command", "Get-WmiObject Win32_Process | Select-Object Caption, ParentProcessId, ProcessId"]).decode().split("\r\n"),

                    #subprocess.check_output("wmic process get Caption,ParentProcessId,ProcessId").decode().split("\r\n"),
                )
            )
            for process in tm_processes:
                name, parent_id, process_id = process.split()
                parent_id = int(parent_id)
                process_id = int(process_id)
                if parent_id == tmi_process_id:
                    self.tm_process_id = process_id
                    print(f"Found Trackmania process id: {self.tm_process_id=}")
                    return

            if time.time() - start_time > 10:
                raise ProcessLookupError("Could not find process after more than 10s of searching.")

    def is_game_running(self) -> bool:
        return (self.tm_process_id is not None) and (self.tm_process_id in (p.pid for p in psutil.process_iter()))

    def launch_game(self):
        """Launches game. Sets process and window-id for tmi."""
        self.tm_process_id = None
        self._run_launchstring_and_set_process_id()

        self.last_game_reboot = time.perf_counter()
        self.latest_map_path_requested = -1
        self.msgtype_response_to_wakeup_TMI = None
        while not self.is_game_running():
            time.sleep(0)

        self._get_tm_window_id()

    def close_game(self):
        """Kills game-process. Is blockig as long as self.is_game_running() is true."""
        self.timeout_has_been_set = False
        self.game_activated = False
        assert self.tm_process_id is not None
        os.system(f"taskkill /PID {self.tm_process_id} /f")
        while self.is_game_running():
            time.sleep(0.1)


    """def request_map(self, map_path: str, zone_centers: npt.NDArray):
        self.latest_map_path_requested = map_path
        map_path = map_path.replace("/", "\\")
        map_loader.hide_personal_record_replay(map_path, True)
        self.iface.execute_command(f"map {map_path}")
        self.UI_disabled = False
        (
            self.next_real_checkpoint_positions,
            self.max_allowable_distance_to_real_checkpoint,
        ) = map_loader.sync_virtual_and_real_checkpoints(zone_centers, map_path)"""

if __name__ == "__main__":

    gmi = GameInstanceManager(Path(os.path.expanduser("~")) / "AppData" / "Local" / "TMLoader" / "TMLoader.exe",
                              Path(os.path.expanduser("~")) / "OneDrive" / "Dokumente" / "TMInterface" /"Plugins" / "Python_Link.as",
                              "default")
    gmi.launch_game()