
import gymnasium as gym
import numpy as np
import time
import logging
import traceback

from typing import Optional
from configs.config import TrainConfig

from trackmania_gym.trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
from trackmania_gym.game_interaction.ipc_command_sender import TMICommunicationFaildException, IPCommandSender
from trackmania_gym.game_interaction.process_management import ProcessManagement
from trackmania_gym.trackmania_env.envs.enivonrments import get_environment

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

    def __init__(self, train_cfg : TrainConfig, port : int = 8775, return_obs_as_dict : bool = True, 
                 lock = None, skip : int = 0, suffix : str = "", track : str = None):
        """
        Args:
            train_cfg (TrainConfig) : Configuration file used for initialization of enviroment
            port (int)              : TCP-port id for communication between TMInterface and ProcessWrapper
            skip (int)              : In case of crash initiailize to port = port + skip
            suffix (str)            : Suffix to append to the logging name. Defaults to no suffix, i.e. ''
            track (str)             : Track that is passed to the process-wrapper instance. Defaults to None and then process-wrapper uses cfg.gmi.track
        """
        super().__init__()

        self.cfg : TrainConfig = train_cfg
        self.port : int = port
        self.skip = skip
        self.env : TMNF_Single_Agent_Env = None

        self.obs_manager = None
        self.rew_calculator = None
        self.reference_line = None
        self.termination_manager = None

        self.env_initalizations : int = 0
        self.queue_empty_error  : int = 0

        self.return_obs_as_dict = return_obs_as_dict

        self.total_timesteps = 0
        self._step_recorded_at_timestep = -1
        self._last_obs, self._last_rew, self._last_terminated, self._last_truncated, self._last_info = None, None, None, None, None
        self.logger = logging.getLogger(self.__class__.__name__ + suffix)
        self.logger.info(f"Initialized with port {self.port}")

        self.lock = lock 
        self.track = track

        self.pm = ProcessManagement(train_config=self.cfg, image_width=self.cfg.rl_env.obs_manager.img_width, image_height=self.cfg.rl_env.obs_manager.img_height, 
                                        port=self.port, lock = self.lock)
        self.ipcsender : IPCommandSender = None

        self.reinit_recursion_depth = 0
        
        
    def _set_env_variables(self):
        self.obs_manager = self.env.obs_manager
        self.rew_calculator = self.env.rew_calculator
        self.reference_line = self.env.reference_line
        self.termination_manager = self.env.termination_manager

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
        
        self.env_initalizations += 1
        self.logger.info(f"Initializing environment for the {self.env_initalizations}-th time.")
        
        self.pm = ProcessManagement(train_config=self.cfg, image_width=self.cfg.rl_env.obs_manager.img_width, image_height=self.cfg.rl_env.obs_manager.img_height, port=self.port, lock = self.lock)
        
        self.ipcsender = self.pm.start_process_and_wait_for_startsignal(track = self.track)

        self.env = get_environment(self.cfg, self.ipcsender)
        self.env.obs_manager.return_as_dict = self.return_obs_as_dict
        self._set_env_variables()

    def finalize_process(self, reinit : bool = True):
        """
        Sends command to process-wrapper to stop execution.
        Args:
            reinit (bool) : If set, it calls self.init_environment() after joining process.
        """
        if self.reinit_recursion_depth > 10:
            self.logger.error("Tried reinitializing environment over 10 times. Killing now.")
            raise RuntimeError("Tried reinitializing environment over 10 times. Killing now.")

        try:
            self.logger.info("Finalizing tmi-process.")
            self.pm.finalize_processes()
        except Exception as e:
            self.logger.exception("Got exception when trying to call self.pm.finalize_processes().")

        time.sleep(10)

        if reinit:
            self.reinit_recursion_depth += 1
            self.port = self.port + self.skip
            self.logger.info(f"Increased port to {self.port}")
            self.queue_empty_error += 1
            self.logger.error(f"Reinitializing environment after queue empty error. Total n errors : {self.queue_empty_error}")
            self.init_environment()
            self.reset()
            self.reinit_recursion_depth = 0

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
