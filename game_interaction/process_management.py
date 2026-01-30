import logging

from game_interaction.process_wrapper import TMIProcessWrapper
from game_interaction.game_instance_manager2 import GameInstanceManager
from game_interaction.ipc_fields import IPCCommands, IPCFields
from game_interaction.ipc_command_sender import IPCommandSender

from multiprocessing import Process, Queue
from configs.config import TrainConfig, EnvConfig

def run_wrapper(gmi, launch_game : bool, cmd_q : Queue, res_q : Queue, track : str, img_w : int, img_h : int, camera_id:int ,env_cfg : EnvConfig, debug: bool): # apparently its better to run process like this to avoid pickel issues or smth?
    """
    Method to run in the process-Wrapper. For arguments @see TMIProcessWrapper-constructor.
    """
    wrapper = TMIProcessWrapper(gmi, launch_game=launch_game, command_queue=cmd_q, 
                                response_queue=res_q, track=track, 
                                img_width=img_w, img_height=img_h,
                                camera_id=camera_id,
                                automatic_prevent_sim_finish= env_cfg.automatic_prevent_sim_finish,
                                actions_per_second=  env_cfg.actions_per_second, 
                                use_rewind= env_cfg.use_rewind,
                                disable_waitforstep_after_n_consecutive_timeouts=  env_cfg.disable_waitforstep_after_n_consecutive_timeouts,
                                debug= debug,
                            )
    wrapper.start_running()


class ProcessManagement:

    def __init__(self, train_config : TrainConfig, image_width : int, image_height : int,lock: str = None, port : int = 8775):
        """
        Args:
            train_config (TrainConfig)        : Configuration for instanciation of GameInstanceManagers
            image_width, image_heihgt (int)   : width and height of requested images.
        """
        self.game_instance_manager = GameInstanceManager.get_instance(
            TMLoader_path = train_config.platforms.tmloader,
            path_to_plugin = train_config.platforms.plugin,
            TMLoader_profile_name= train_config.gmi.tm_loader_profile_name,
            linux = train_config.platforms.os == "linux",
            headless= train_config.gmi.headless,
            tmi_port= port,
            lock=lock,)

        self.train_config = train_config
        self.image_width = image_width
        self.image_height = image_height
        self.lock = lock
        self.port = port

        self.control_queue = Queue() # queue for commands to send to TMIProcessWrapper
        self.response_queue = Queue() # answers (payload) from TMIProcess Wrapper

        self.tmi_process : Process = None
        self.trackmania_pid : int = 0
        self.logger = logging.getLogger(self.__class__.__name__)

        self.sender = None

    def wait_for_startsignal(self) -> None:
        """wait for trackmania to load map and start simulation."""
        startsignal : dict = self.sender.send_command(IPCCommands.get_startsignal())
        assert startsignal[IPCFields.STATUS] == IPCFields.STATUS_OK
        
        if IPCFields.ARGS in startsignal:
            self.trackmania_pid = startsignal[IPCFields.ARGS]["tm_pid"]
        else:
            self.logger.error(f"Expected the startsignal to contain ARGS, but thats not the case, its only {startsignal}")
    
    def start_process_and_wait_for_startsignal(self, timeout: float = 15, mcs: int = 2, wfr: float = 0.5, track : str = None) -> IPCommandSender:
        """Starts a process that wrapps TMIProcessWrapper to communicate with Trackmania instance. Waits for established communication after launching process.
        If no connection can be found or no answer received, Queue.Empty-Error is thrown.
        Args:
            - timeout (float)   : timeout passed onto the sender
            - mcs   (int)       : mcs passed to sender
            - wfr   (float)     : wfr passed to sender
            - track (str)       : String specifiying track-file. Defaults to None and then uses self.train_config.gmi.track
        
        
        Returns:
            - sender (IPCommandSender)  : Sending module to communicate with the process-wrapper, it managees: 
                - cq : control-Queue to be able to send commands to the running process @see TMIProcessWrapper for format
                - rq : response-Queue to be able to receive answers from running process @see TMIProcessWrapper for format (depends on command.)
        """
        defaulttrack = self.train_config.gmi.track if track is None else track
        p = Process(target=run_wrapper, 
                    args=(self.game_instance_manager,
                        self.train_config.gmi.launch,
                        self.control_queue,
                        self.response_queue,
                        defaulttrack,
                        self.image_width,
                        self.image_height,
                        self.train_config.rl_env.obs_manager.camera_id,
                        self.train_config.rl_env.env,
                        self.train_config.debug,))
        self.tmi_process = p
        p.start()
        self.sender = IPCommandSender(self.control_queue, self.response_queue, timeout=timeout, mcs = mcs, wfr = wfr)

        self.wait_for_startsignal()
        return self.sender


    def finalize_processes(self):
        """Finalizes Process-Wrapper and also sends additional pkill-commands to make sure tmnf and process-wrapper have been killed."""
        
        if self.sender.useable:
            try:
                self.sender.send_command(IPCCommands.get_end_syncloop_command())
            except Exception as e:
                self.logger.exception("Could not send killcommanmd to process-wrapper.")
        self.tmi_process.join()
        self.game_instance_manager.tm_process_id = self.trackmania_pid #super duper ugly bugly
        self.game_instance_manager.close_game()
        
        self.game_instance_manager.tm_process_id = self.tmi_process.pid
        self.game_instance_manager.close_game()
