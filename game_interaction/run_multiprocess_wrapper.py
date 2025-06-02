

from game_interaction.process_wrapper import TMIProcessWrapper
from multiprocessing import Process, Queue


def run_wrapper(gmi, launch_game : bool, cmd_q : Queue, res_q : Queue, img_w : int, img_h : int): # apparently its better to run process like this to avoid pickel issues or smth?
    """
    Method to run in the process
    """
    wrapper = TMIProcessWrapper(gmi, launch_game=launch_game, command_queue=cmd_q, response_queue=res_q, img_width=img_w, img_height=img_h)
    wrapper.syncloop()


def start_process_and_wait_for_startsignal(GIM, launch : bool, image_width : int, image_height : int) -> tuple[Process, Queue, Queue]:
    """Starts a process that wrapps TMIProcessWrapper to communicate with Trackmania instance. Waits for established communication after launching process.
    If no connection can be found or no answer received, Queue.Empty-Error is thrown.
    
    -- GIM : GameInstanceManager 
    -- launch : If True (should be True), tells GIM to launch the game after process is created (this prevents timeout from launching game to waiting for server response)
    -- image_width, image_heihgt : width and height of requested images.
    
    
    Returns [p, cq, rq]
    --------
    - p : Process that runs the communication
    - cq : control-Queue to be able to send commands to the running process @see TMIProcessWrapper for format
    - rq : response-Queue to be able to receive answers from running process @see TMIProcessWrapper for format (depends on command.)"""

    control_queue = Queue() # queue for commands to send to TMIProcessWrapper
    response_queue = Queue() # answers (payload) from TMIProcess Wrapper
   
    p = Process(target=run_wrapper, args=(GIM, launch, control_queue, response_queue, image_width, image_height))
    
    p.start()

    # wait for trackmania to load map and start simulation.
    control_queue.put_nowait(TMIProcessWrapper.IPCCommands.get_startsignal(512))
    startsignal = response_queue.get(timeout = 60)
    assert startsignal["cmd_id"] == 512 and startsignal["status"] == 0

    return p, control_queue, response_queue