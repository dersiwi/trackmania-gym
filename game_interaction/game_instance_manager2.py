from __future__ import annotations

import subprocess
import os
import time 
import psutil
import win32process
import win32gui 
from pathlib import Path
from multiprocessing.synchronize import Lock

from game_interaction.tminterface2 import TMInterface

class GameInstanceManager:
    


    @staticmethod
    def get_instance(TMLoader_path : str, path_to_plugin : str, TMLoader_profile_name : str = "default", linux : bool = False) -> GameInstanceManager:
        """
        The GameInstanceManager launches the game from the operating systems side via a system command (launch_game() and close_game() start and end tmnf processes.)
        To get an instance of the GameInstanceManager use this method and specify the operating system by setting linux accordingly.

        The GameInstanceManager also sets instanciates a TMInterface, which can be accessed directly or via get_tminterface().


        params
        ------

        - TMLoader_path : path to the TMLoader executable
        - TMLoader_profile_name : name of the profile inside the TMLoader to be used; if none is specifeid the "default"-profile is used
        - path_to_plugin : path to the plugin aka. Python_Link.as that should be placed inside the trackmania-plugin folder
        - linux : set to true if on a linux operating system, set to false if on a windows operating system
        """

        if linux:
            # TODO get Lock.
            return GameInstanceMangerLinux(TMLoader_path, TMLoader_profile_name, path_to_plugin, None)
        else:
            return GameInstanceManagerWindows(TMLoader_path, TMLoader_profile_name, path_to_plugin)


    def __init__(self, TMLoader_path : str, TMLoader_profile_name : str, path_to_plugin : str):
        """Do not use this class directly. Instanciate via GameInstanceManager.get_instance()."""

        self.TMLoader_path : str = TMLoader_path
        self.TMLoader_profile_name = TMLoader_profile_name
        self.path_to_plugin = path_to_plugin
        self.tmi_port = 8775
        self.tm_process_id = None
        self.tm_window_id  = None
        self.tminterface = TMInterface(self.tmi_port)

        assert os.path.exists(self.path_to_plugin), f"Python_Link.as was not found at '{self.path_to_plugin}'."


    def get_tminterface(self) -> TMInterface:
        """Get the TMInterface instance associated with this game-instance."""
        return self.tminterface

    def register_iface(self, timeout_in_s : int = 10) -> None:
        """Calls self.tminterface.register(timeout). Is blocking as long as tminterface is not registered or ConnectionRefusedError."""
        if not self.tminterface.registered:
            while True:
                try:
                    self.tminterface.register(10)
                    break
                except ConnectionRefusedError as e:
                    print(e)

    def is_game_running(self) -> bool:
        return (self.tm_process_id is not None) and (self.tm_process_id in (p.pid for p in psutil.process_iter()))
    
    def _get_gameprocess_killcommand(self) -> str:
        """Returns correct killcommand for the process according to the operating system."""
        raise NotImplementedError("Do not use this class directly.")
    
    def launch_game(self) -> None:
        """Launches a tmnf process. Also sets self.tm_process_id and self.tm_window_id."""
        raise NotImplementedError()
    
    def close_game(self) -> None:
        """Kills tmnf process according to set self.tm_process_id. self.launch_game() has to be called beforehand. """
        self.timeout_has_been_set = False
        self.game_activated = False
        assert self.tm_process_id is not None
        os.system(self._get_gameprocess_killcommand())
        while self.is_game_running():
            time.sleep(0.1)
    

class GameInstanceManagerWindows(GameInstanceManager):

    def __init__(self, TMLoader_path, TMLoader_profile_name, path_to_plugin):
        super().__init__(TMLoader_path, TMLoader_profile_name, path_to_plugin)

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

    def launch_game(self):
        self.tm_process_id = None
        self._run_launchstring_and_set_process_id()

        self.last_game_reboot = time.perf_counter()
        self.latest_map_path_requested = -1
        self.msgtype_response_to_wakeup_TMI = None
        while not self.is_game_running():
            time.sleep(0)

        self._get_tm_window_id()


    def _get_gameprocess_killcommand(self) -> str:
        return f"taskkill /PID {self.tm_process_id} /f"

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


class GameInstanceMangerLinux(GameInstanceManager):

    def __init__(self, TMLoader_path, TMLoader_profile_name, path_to_plugin, game_spawning_lock : Lock):
        from xdo import Xdo
        super().__init__(TMLoader_path, TMLoader_profile_name, path_to_plugin)
        self.game_spawning_lock : Lock = game_spawning_lock

    def _get_tm_window_id(self):
        self.tm_window_id = None
        while self.tm_window_id is None:  # This outer while is for the edge case where the window may not have had time to be launched
            window_search_depth = 1
            while True:  # This inner while is to try and find the right depth of the window in Xdo().search_windows()
                c1 = set(Xdo().search_windows(winname=b"TrackMania Modded", max_depth=window_search_depth + 1))
                c2 = set(Xdo().search_windows(winname=b"TrackMania Modded", max_depth=window_search_depth))
                c1 = {w_id for w_id in c1 if Xdo().get_pid_window(w_id) == self.tm_process_id}
                c2 = {w_id for w_id in c2 if Xdo().get_pid_window(w_id) == self.tm_process_id}
                c1_diff_c2 = c1.difference(c2)
                if len(c1_diff_c2) == 1:
                    self.tm_window_id = c1_diff_c2.pop()
                    break
                elif (
                    len(c1_diff_c2) == 0 and len(c1) > 0
                ) or window_search_depth >= 10:  # 10 is an arbitrary cutoff in this search we do not fully understand
                    print(
                        "Warning: Worker could not find the window of the game it just launched, stopped at window_search_depth",
                        window_search_depth,
                    )
                    break
                window_search_depth += 1

    def _get_tm_pids(self) -> list[int]:
        return [process.pid for process in psutil.process_iter() if self._is_tm_process(process)]
    
    def _is_tm_process(self, process: psutil.Process) -> bool:
        try:
            return process.name().startswith("TmForever")
        except psutil.NoSuchProcess:
            return False
    
    def launch_game(self):
        self.game_spawning_lock.acquire()
        pid_before = self._get_tm_pids()
        os.system(str(self.TMLoader_path) + " " + str(self.tmi_port))
        while True:
            pid_after = self._get_tm_pids()
            tmi_pid_candidates = set(pid_after) - set(pid_before)
            if len(tmi_pid_candidates) > 0:
                assert len(tmi_pid_candidates) == 1
                break
        self.tm_process_id = list(tmi_pid_candidates)[0]

    def _get_gameprocess_killcommand(self) -> str:
        return "kill -9 " + str(self.tm_process_id)

if __name__ == "__main__":

    gmi = GameInstanceManager(Path(os.path.expanduser("~")) / "AppData" / "Local" / "TMLoader" / "TMLoader.exe",
                              Path(os.path.expanduser("~")) / "OneDrive" / "Dokumente" / "TMInterface" /"Plugins" / "Python_Link.as",
                              "default")
    gmi.launch_game()