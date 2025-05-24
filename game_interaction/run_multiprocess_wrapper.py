

from game_interaction.process_wrapper import TMIProcessWrapper
from multiprocessing import Process, Queue


def run_wrapper(iface, cmd_q : Queue, res_q : Queue, img_w : int, img_h : int): # apparently its better to run process like this to avoid pickel issues or smth?
    """
    Method to run in the process
    """
    wrapper = TMIProcessWrapper(iface, command_queue=cmd_q, response_queue=res_q, img_width=img_w, img_height=img_h)
    wrapper.syncloop()
