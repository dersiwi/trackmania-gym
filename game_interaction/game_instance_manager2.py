"""
This module includes code adapted and refactored from the Linesight-AI project
(https://github.com/Linesight-RL/linesight). Credit to the original authors
for the foundational implementation; refactloring changes made here.
"""

from __future__ import annotations

import subprocess
import os
import time 
from filelock import FileLock
import psutil
import logging
if os.name == 'nt': 
    import win32process
    import win32gui 
    import win32.lib.win32con as win32con
    import win32com.client
else:
    from xdo import Xdo
    from Xlib import X
from pathlib import Path


from multiprocessing import Lock
from game_interaction.tminterface2 import TMInterface

class GameInstanceManager:
    


    @staticmethod
    def get_instance(TMLoader_path : str, path_to_plugin : str, TMLoader_profile_name : str = "default", linux : bool = False, headless : bool = False,tmi_port:int = 8775,lock = None, wineprefix : str = None) -> GameInstanceManager:
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
        - headless : if True, starts the game headless (i.e. with virtual monitor) - as of now this is highly experimental and only has an effect on linux.
        """

        if linux:
            # TODO get Lock.
            return GameInstanceMangerLinux(TMLoader_path, TMLoader_profile_name, path_to_plugin, lock, headless,tmi_port, wineprefix=wineprefix)
        else:
            return GameInstanceManagerWindows(TMLoader_path, TMLoader_profile_name, path_to_plugin, headless,tmi_port)


    def __init__(self, TMLoader_path : str, TMLoader_profile_name : str, path_to_plugin : str, headless : bool,tmi_port:int):
        """Do not use this class directly. Instanciate via GameInstanceManager.get_instance()."""

        self.TMLoader_path : str = TMLoader_path
        self.TMLoader_profile_name : str= TMLoader_profile_name
        self.path_to_plugin : str = path_to_plugin
        self.headless : bool = headless
        self.tmi_port = tmi_port
        self.tm_process_id = None
        self.tm_window_id  = None
        self.tminterface = TMInterface(self.tmi_port)
        self.game_activated = False

        assert os.path.exists(self.path_to_plugin), f"Python_Link.as was not found at '{self.path_to_plugin}'."
        self.logger = logging.getLogger(self.__class__.__name__)

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
                    self.logger.error(e)
                    time.sleep(0.5)

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
    
    def _set_window_focus(self):
        """Sets focus on the specified game window ."""
        raise NotImplementedError()

class GameInstanceManagerWindows(GameInstanceManager):

    def __init__(self, TMLoader_path, TMLoader_profile_name, path_to_plugin, headless,tmi_port):
        super().__init__(TMLoader_path, TMLoader_profile_name, path_to_plugin, headless,tmi_port)

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
                    self.logger.info(f"Found Trackmania window id: {self.tm_window_id=}")
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

        
        output = subprocess.check_output(self.__get_launch_string()).decode()
        tmi_process_id = int(output.split("\r\n")[1].strip())

        try:
            loader_proc = psutil.Process(tmi_process_id)
        except psutil.NoSuchProcess:
            raise ProcessLookupError(f"TMLoader process {tmi_process_id} exited too early")

        start_time = time.time()
        while True:
            children = loader_proc.children(recursive=True)
            for child in children:
                if "TmForever" in child.name():
                    self.tm_process_id = child.pid
                    self.logger.info(f"Found Trackmania process id: {self.tm_process_id}")
                    return

            if not loader_proc.is_running():
                raise ProcessLookupError("TMLoader exited before TmForever started.")

            if time.time() - start_time > 10:
                raise ProcessLookupError("Could not find Trackmania process after 10s.")

            time.sleep(0.5)

    def launch_game(self):
        self.tm_process_id = None
        self._run_launchstring_and_set_process_id()

        self.last_game_reboot = time.perf_counter()
        self.latest_map_path_requested = -1
        self.msgtype_response_to_wakeup_TMI = None
        while not self.is_game_running():
            time.sleep(0.1)

        self._get_tm_window_id()


    def _get_gameprocess_killcommand(self) -> str:
        return f"taskkill /PID {self.tm_process_id} /f"
    
    def _set_window_focus(self):
        if not self.game_activated:
            shell = win32com.client.Dispatch("WScript.Shell")
            shell.SendKeys("%")
            win32gui.SetForegroundWindow(self.tm_window_id)
            self.game_activated = True


class GameInstanceMangerLinux(GameInstanceManager):

    launched_xvfb : bool = False
    xvfb_launch_dict : dict[str, str] = None

    @staticmethod
    def launch_xvfb(display_number : str = ":99", screen_number : str = "0", resolution : str = "1280x720x24") -> None:
        """launches a virtual framebuffer process. And sets xvfb_launch_dict. 
        
        params
        ------
        - display_number : defines the number of the display
        - screen_number : forms the full identifier for the display (display_number:screen_number, default : 99:0)
        - resolution : resolution of the display in "[width]x[height]x[colordepth_in_bit]" (24 --> RGB, 8 bit for each channel)

                
        """
        if GameInstanceMangerLinux.launched_xvfb: # TODO : does this make sense? But i think we only need one virtual display.
            return 
        xvfb_proc = subprocess.Popen(['Xvfb', display_number, '-screen', screen_number, resolution])
        time.sleep(1)
        GameInstanceMangerLinux.launched_xvfb = True
        GameInstanceMangerLinux.xvfb_launch_dict = {"DISPLAY" : display_number}
        print("Launched xvfb-process.")


    def __init__(self, TMLoader_path, TMLoader_profile_name, path_to_plugin, game_spawning_lock : None, headless : bool,tmi_port:int, wineprefix : str):
        super().__init__(TMLoader_path, TMLoader_profile_name, path_to_plugin, headless,tmi_port)
        self.game_spawning_lock : str = game_spawning_lock
        self.wineprefix = wineprefix


    def _get_tm_window_id(self):
        from xdo import Xdo
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
                    self.logger.warning(
                        "Worker could not find the window of the game it just launched, stopped at window_search_depth",
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

    def __get_tmnf_process_id(self, pid_before , timeout: int = 10):
        launch_time = time.time()
        while True:
            time.sleep(0.25)
            pid_after = set(self._get_tm_pids())
            new_pids = pid_after - pid_before
            if new_pids:
                if len(new_pids) == 1:
                    self.tm_process_id = new_pids.pop()
                    break
                else:
                    self.logger.warning(f"Multiple new PIDs detected: {new_pids}")
                    self.tm_process_id = list(new_pids)[0]  # just pick the first one?
                    break

            if time.time() - launch_time > timeout:
                raise TimeoutError(f"TMNF process did not launch within {timeout} seconds.")

    def __get_launch_cmds(self):
        return [
            "wine",
            str(self.TMLoader_path),
            "run",
            "TmForever",
            str(self.TMLoader_profile_name),
            f"/configstring=set custom_port {self.tmi_port}",
        ]
        
    def launch_game(self, timeout=10):
        """Launches the game with timeout 10s to find process ids."""

        if self.game_spawning_lock: self.game_spawning_lock.acquire()

        pid_before = set(self._get_tm_pids())
        launch_cmds = self.__get_launch_cmds()
        
        env = os.environ.copy()
        if not self.wineprefix is None:
            env["WINEPREFIX"] = self.wineprefix
        if self.headless:
            GameInstanceMangerLinux.launch_xvfb()
            process = subprocess.Popen(launch_cmds, env=GameInstanceMangerLinux.xvfb_launch_dict)
        else:
            process = subprocess.Popen(launch_cmds, env=env)

        _ = psutil.Process(process.pid)
        self.__get_tmnf_process_id(pid_before, timeout)
        self._get_tm_window_id()
        if self.game_spawning_lock: self.game_spawning_lock.release()
        return self.tm_process_id

    def _get_gameprocess_killcommand(self) -> str:
        return "kill -9 " + str(self.tm_process_id)
    
    def _set_window_focus(self):
        if not self.game_activated:
            xdo_instance = Xdo() 
            self.logger.info(f"[{os.getpid()}] Waiting for window {self.tm_window_id} to be viewable...")
            xdo_instance.wait_for_window_map_state(self.tm_window_id, X.IsViewable)
            self.logger.info(f"[{os.getpid()}] Window {self.tm_window_id} is now viewable.")
            self.logger.info(f"DEBUG: tm_window_id type: {type(self.tm_window_id)}")
            self.logger.info(f"DEBUG: tm_window_id value: {self.tm_window_id}")
            xdo_instance.activate_window(self.tm_window_id)
            xdo_instance.raise_window(self.tm_window_id)
            self.game_activated= True

if __name__ == "__main__":

    gmi = GameInstanceManager(Path(os.path.expanduser("~")) / "AppData" / "Local" / "TMLoader" / "TMLoader.exe",
                              Path(os.path.expanduser("~")) / "OneDrive" / "Dokumente" / "TMInterface" /"Plugins" / "Python_Link.as",
                              "default")
    gmi.launch_game()
