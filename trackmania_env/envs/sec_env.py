
import gymnasium as gym
import numpy as np
import time
import logging
import traceback

from typing import Optional
from multiprocessing import Process, Queue


from configs.config import TrainConfig

from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env, TMICommunicationFaildException

from game_interaction.run_multiprocess_wrapper import start_process_and_wait_for_startsignal
from game_interaction.ipc_fields import IPCCommands



class CrashProofEnvironment(gym.Env):
    """
    This class implements a wrapper for TMNF_Single_Agent_Env. During training and testing we experienced weird behaviour
    of the process wrapper; after a seemingly random prevent-sim-finish the process wrapper would not answer the commands of 
    the environment anymore. Even sending the command multiple times did not recover the state.

    This class is able to detect the error and if it catches the error, it restarts the enviornment and game completely. This enables the
    learner process using this class to have a clearn interface.

    If the error is detected, no further informaiton from the current enviornment can be gathered. Therefore, the last recorded state is sent back
    once more; however truncated is set to True. The learner process can then call reset, in order to keep disruption to trajectory collection mininmal.   
    """
    _env = None
    @staticmethod
    def get_instance(env_id : int = 0) -> TMNF_Single_Agent_Env:
        assert CrashProofEnvironment._env is not None, "Cannot return instance of environment, as is none. Call init_environment beforehadn."
        return CrashProofEnvironment._env


    def __init__(self, train_cfg : TrainConfig):
        """
        Args:
            train_cfg (TrainConfig) : Configuration file used for initialization of enviroment  
        """
        super().__init__()

        self.cfg : TrainConfig = train_cfg
        self.env : TMNF_Single_Agent_Env = None

        self.tmi_process : Process  = None
        self.control_queue : Queue  = None
        self.response_queue : Queue = None

        self.env_initalizations : int = 0
        self.queue_empty_error  : int = 0

        self.total_timesteps = 0
        self._step_recorded_at_timestep = -1
        self._last_obs, self._last_rew, self._last_terminated, self._last_truncated, self._last_info = None, None, None, None, None
        self.logger = logging.getLogger(self.__class__.__name__)
        

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space


    def init_environment(self) -> None:
        """Initializes the environment according to the config file given at initializaiton:
            1) start_process_and_wait_for_startsignal(...)
            2) get_environment(...)
        This enviornment is then used for further interaction"""
        from trackmania_env.envs.enivonrments import get_environment    # TODO put this back at the top.
        self.env_initalizations += 1
        self.logger.info(f"Initializing environment for the {self.env_initalizations}-th time.")
        self.tmi_process, self.control_queue, self.response_queue = start_process_and_wait_for_startsignal(self.cfg, 
                                                                                            self.cfg.rl_env.obs_manager.img_width, 
                                                                                            self.cfg.rl_env.obs_manager.img_height)
        self.env = get_environment(self.cfg, self.control_queue, self.response_queue)
        CrashProofEnvironment._env = self.env.env

    def finalize_process(self, reinit : bool = True):
        """
        Sends command to process-wrapper to stop execution.
        Args:
            reinit (bool) : If set, it calls self.init_environment() after joining process.
        """
        self.logger.info("Finalizing tmi-process.")

        self.control_queue.put(IPCCommands.get_end_syncloop_command(1000)) #1000 doesnt matter.
        self.tmi_process.join()
        time.sleep(10)

        if reinit:
            self.queue_empty_error += 1
            self.logger.error(f"Reinitializing environment after queue empty error. Total n errors : {self.queue_empty_error}")
            self.init_environment()
            self.env.reset()

    def _answer_step_after_crash(self, action) -> tuple[gym.spaces.Dict,float,bool,bool, dict[str, any]]:
        if self.total_timesteps == 0:
            self.step(action)
        return self._last_obs, self._last_rew, self._last_terminated, True, self._last_info

    def step(self, action) -> tuple[gym.spaces.Dict,float,bool,bool, dict[str, any]]:
        try:
            self._last_obs, self._last_rew, self._last_terminated, self._last_truncated, self._last_info = self.env.step(action)
            self.total_timesteps += 1
            self._step_recorded_at_timestep = self.total_timesteps
            return self._last_obs, self._last_rew, self._last_terminated, self._last_truncated, self._last_info
        except TMICommunicationFaildException as rte: # queue-empty error
            traceback.print_exc()
            self.finalize_process()
            return self._answer_step_after_crash(action)




    def reset(self, seed = None, options = None) -> tuple[gym.spaces.Dict, dict[str, any]]:
        try:
            return self.env.reset(seed = seed, options = options)
        except TMICommunicationFaildException as rte: # queue-empty error
            traceback.print_exc()
            self.finalize_process()
            return self.reset(seed = seed, options = options)


    def render(self, mode: str = "human") -> Optional[np.array]:
        try:
            return self.env.render(mode=mode)
        except TMICommunicationFaildException as rte: # queue-empty error
            traceback.print_exc()
            self.finalize_process()
            return self.render(mode=mode)