
import gymnasium as gym
import numpy as np
import time
import logging
import traceback

from typing import Optional
from multiprocessing import Process, Queue


from configs.config import TrainConfig

from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env, TMICommunicationFaildException
from trackmania_env.envs.enivonrments import get_environment

from game_interaction.run_multiprocess_wrapper import start_process_and_wait_for_startsignal
from game_interaction.ipc_fields import IPCCommands



class CrashProofEnvironment(gym.Env):


    def __init__(self, train_cfg : TrainConfig):
        super().__init__()

        self.cfg : TrainConfig = train_cfg
        self.env : TMNF_Single_Agent_Env = None

        self.tmi_process : Process  = None
        self.control_queue : Queue  = None
        self.response_queue : Queue = None

        self.env_initalizations : int = 0
        self.queue_empty_error  : int = 0
        self.logger = logging.getLogger(self.__class__.__name__)
        

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space


    def init_environment(self):
        self.env_initalizations += 1
        self.logger.info(f"Initializing environment for the {self.env_initalizations}-th time.")
        self.tmi_process, self.control_queue, self.response_queue = start_process_and_wait_for_startsignal(self.cfg, 
                                                                                            self.cfg.rl_env.obs_manager.img_width, 
                                                                                            self.cfg.rl_env.obs_manager.img_height)
        self.env = get_environment(self.cfg, self.control_queue, self.response_queue)


    def finalize_process(self, reinit : bool = True):
        self.logger.info("Finalizing tmi-process.")

        self.control_queue.put(IPCCommands.get_end_syncloop_command(1000)) #1000 doesnt matter.
        self.tmi_process.join()
        time.sleep(10)

        if reinit:
            self.queue_empty_error += 1
            self.logger.error(f"Reinitializing environment after queue empty error. Total n errors : {self.queue_empty_error}")
            self.env.reset()
            self.init_environment()



    def step(self, action) -> tuple[gym.spaces.Dict,float,bool,bool, dict[str, any]]:
        try:
            return self.env.step(action)
        except TMICommunicationFaildException as rte: # queue-empty error
            traceback.print_exc()
            self.finalize_process()
            return self.step(action)

    def reset(self, seed = None, options = None) -> tuple[gym.spaces.Dict, dict[str, any]]:
        try:
            return self.env.reset(seed = seed, options = options)
        except TMICommunicationFaildException as rte: # queue-empty error
            traceback.print_exc()
            self.finalize_process()
            return self.reset()


    def render(self, mode: str = "human") -> Optional[np.array]:
        try:
            return self.env.render(mode=mode)
        except TMICommunicationFaildException as rte: # queue-empty error
            traceback.print_exc()
            self.finalize_process()
            return self.render()