

from game_interaction.process_wrapper import TMIProcessWrapper
from game_interaction.game_instance_manager2 import GameInstanceManager

from multiprocessing import Process, Queue
from configs.config import TrainConfig, GMIConfig, PlatformConfig

def run_wrapper(gmi, launch_game : bool, cmd_q : Queue, res_q : Queue, track : str, img_w : int, img_h : int, automatic_prevent_sim_finish : bool): # apparently its better to run process like this to avoid pickel issues or smth?
    """
    Method to run in the process-Wrapper. For arguments @see TMIProcessWrapper-constructor.
    """
    wrapper = TMIProcessWrapper(gmi, launch_game=launch_game, command_queue=cmd_q, 
                                response_queue=res_q, track=track, 
                                img_width=img_w, img_height=img_h, 
                                automatic_prevent_sim_finish = automatic_prevent_sim_finish)
    wrapper.syncloop()


def start_process_and_wait_for_startsignal(platform : PlatformConfig, gmi_cfg : GMIConfig, image_width : int, image_height : int) -> tuple[Process, Queue, Queue]:
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
        TMLoader_path = platform.tmloader,
        path_to_plugin = platform.plugin,
        TMLoader_profile_name= gmi_cfg.tm_loader_profile_name,
        linux = platform.os == "linux",
        headless= gmi_cfg.headless)

    control_queue = Queue() # queue for commands to send to TMIProcessWrapper
    response_queue = Queue() # answers (payload) from TMIProcess Wrapper
   
    p = Process(target=run_wrapper, args=(GIM, gmi_cfg.launch, control_queue, response_queue, gmi_cfg.track, image_width, image_height, gmi_cfg.automatic_prevent_sim_finish))
    
    p.start()

    # wait for trackmania to load map and start simulation.
    control_queue.put_nowait(TMIProcessWrapper.IPCCommands.get_startsignal(512))
    startsignal = response_queue.get(timeout = 60)
    assert startsignal["cmd_id"] == 512 and startsignal["status"] == 0

    return p, control_queue, response_queue