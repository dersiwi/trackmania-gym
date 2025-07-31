

from game_interaction.process_wrapper import TMIProcessWrapper
from game_interaction.game_instance_manager2 import GameInstanceManager
from game_interaction.ipc_fields import IPCCommands

from multiprocessing import Process, Queue
from configs.config import TrainConfig, GMIConfig, PlatformConfig, EnvConfig

def run_wrapper(gmi, launch_game : bool, cmd_q : Queue, res_q : Queue, track : str, img_w : int, img_h : int, camera_id:int ,env_cfg : EnvConfig): # apparently its better to run process like this to avoid pickel issues or smth?
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
                               )
    wrapper.syncloop()


def start_process_and_wait_for_startsignal(train_config : TrainConfig, image_width : int, image_height : int,lock: str = None) -> tuple[Process, Queue, Queue]:
    """Starts a process that wrapps TMIProcessWrapper to communicate with Trackmania instance. Waits for established communication after launching process.
    If no connection can be found or no answer received, Queue.Empty-Error is thrown.
    
    -- platform : Platform configuration to help instanciate gmi
    -- gmi_cfg : GMI-configuration needed for instanciation
    -- launch : If True (should be True), tells GIM to launch the game after process is created (this prevents timeout from launching game to waiting for server response)
    -- image_width, image_heihgt : width and height of requested images.
    
    
    Returns [p, cq, rq]
    --------
    - p : Process that runs the communication
    - cq : control-Queue to be able to send commands to the running process @see TMIProcessWrapper for format
    - rq : response-Queue to be able to receive answers from running process @see TMIProcessWrapper for format (depends on command.)"""

    GIM = GameInstanceManager.get_instance(
        TMLoader_path = train_config.platforms.tmloader,
        path_to_plugin = train_config.platforms.plugin,
        TMLoader_profile_name= train_config.gmi.tm_loader_profile_name,
        linux = train_config.platforms.os == "linux",
        headless= train_config.gmi.headless,
        tmi_port= train_config.gmi.port,
        lock=lock,)

    control_queue = Queue() # queue for commands to send to TMIProcessWrapper
    response_queue = Queue() # answers (payload) from TMIProcess Wrapper
   
    p = Process(target=run_wrapper, 
                args=(GIM,
                    train_config.gmi.launch,
                    control_queue,
                    response_queue,
                    train_config.gmi.track,
                    image_width,
                    image_height,
                    train_config.rl_env.obs_manager.camera_id,
                    train_config.rl_env.env,
                    ))
    
    p.start()

    # wait for trackmania to load map and start simulation.
    control_queue.put_nowait(IPCCommands.get_startsignal(512))
    startsignal = response_queue.get(timeout = 60)
    assert startsignal["cmd_id"] == 512 and startsignal["status"] == 0

    return p, control_queue, response_queue