

# Hydra related imports
import hydra

# gymnasium environment wrapper 
from gymnasium import ObservationWrapper
from stable_baselines3.common.monitor import Monitor

# extractor imports
from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
from trackmania_env.envs.testenv_single_agent import TestEnvironment

from trackmania_env.utils.reference_line_manager import ReferenceLineManager
from trackmania_env.observations.observations import get_observation_manager
from trackmania_env.rewards.getrewards import get_reward_calculator
from trackmania_env.terminations.get_termination_manager import get_termination_manager

from trackmania_env.utils.orientationless_random_respawn_manager import OrientationlessRespawnManager

from configs.config import TrainConfig
from multiprocessing import Queue
import gymnasium as gym

def get_environment(cfg : TrainConfig, control_queue : Queue, response_queue : Queue, test : bool = False) -> gym.Env:
    """Initializes environment according to given configuration file and applies wrappers, if specified in conifg.
    Parameters
    ----------
        - cfg : Train-Configuration
        - control_queue : Queue used by environment to send controls to Process-Wrapper
        - response_queue : Response Queue used by environment to get responses by ProcessWrapper
        - test (Default : Falsee): In addition to setting test in config, setting this true always returns a test environment and does not apply any wrappers to the environment.
    """

    obs_manager = get_observation_manager(cfg = cfg, wrap_obs_in_test = cfg.rl_env.env.wrap_obs_in_test, normalize=cfg.rl_env.env.normalize_obs)
    reward_calculator = get_reward_calculator(reward_calculator_cfg = cfg.rl_env.reward_manager, normalize=cfg.rl_env.env.normalize_rewards)
    termination_manger = get_termination_manager(termination_cfg= cfg.rl_env.termination_manager)

    if cfg.rl_env.env.test or test:
        TM_ENV_CLASS = TestEnvironment
    else:
        TM_ENV_CLASS = TMNF_Single_Agent_Env

    tm_env = TM_ENV_CLASS(command_queue=control_queue,
                                    response_queue=response_queue, 
                                    obs_manager=obs_manager,
                                    reward_calculator=reward_calculator,
                                    termination_manger=termination_manger,
                                    reference_line = ReferenceLineManager(cfg.gmi.reference_line),
                                    env_cfg=cfg.rl_env.env,
                                    platform= cfg.platforms.os)
    tm_env.orientationless_respawn_manager = OrientationlessRespawnManager(respawn_coordinates=OrientationlessRespawnManager.get_respawns_for_very_long_checkpoints())
    
    reward_calculator.set_env(tm_env)
    obs_manager.set_env(tm_env)
    termination_manger.set_env(tm_env)

    if not test:
        # apply (Observation)-wrappers to the environment : only relevant for training with sb3
        for _, wrapper_conf in cfg.rl_env.wrappers.items():
            wrapper : ObservationWrapper = hydra.utils.instantiate(wrapper_conf)
            print(f"Wrapping environment in {wrapper.__class__.__name__}")
            tm_env = wrapper(env=tm_env)

    return tm_env