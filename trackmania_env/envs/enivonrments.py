from multiprocessing import Queue
import gymnasium as gym


from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env, ContinuousTMNF_Single_Agent_Env,FakeContinuousTMNF_Single_Agent_Env
from trackmania_env.envs.testenv_single_agent import TestEnvironment

from trackmania_env.observations.observations import get_observation_manager_from_cfg
from trackmania_env.rewards.getrewards import get_reward_calculator_from_cfg
from trackmania_env.terminations.get_termination_manager import get_termination_manager

from trackmania_env.utils.orientationless_random_respawn_manager import OrientationlessRespawnManager
from game_interaction.ipc_command_sender import IPCommandSender
from configs.config import TrainConfig



def _get_env_constructor_args(cfg : TrainConfig, ipcommandsender, obs_manager, reward_calculator, termination_manager) -> dict:
    """Creates constructor args that can be passed to the environment
    Args:
        cfg (TrainConfig)               : Configuration used to initialize environment
        control_queue (Queue)           :  used by environment to send controls to Process-Wrapper
        response_queue (Response)       :  Queue used by environment to get responses by ProcessWrapper
        obs_manager (ObservationManager)        : ObservationManager of the environment
        reward_calculator (Rewardcalculator)    : Reward calculator for environment
        termination_manager (Terminationmanager)    : Terminationmanager of the enviornment
    Returns:
        Construtor arguments ready to be passed into environment
    """
    env_cfg = cfg.rl_env.env  
    constructor_kwargs = dict(
        ipcommandsender= ipcommandsender,
        obs_manager= obs_manager,
        reward_calculator= reward_calculator,
        termination_manger= termination_manager,
        track = cfg.gmi.track,
        reset_mode= env_cfg.reset_mode,
        n_previous_actions= env_cfg.n_previous_actions,
        position_buffer_size= env_cfg.position_buffer_size,
        position_moved_threshold= env_cfg.position_moved_threshold,
        ignore_stuck_for_n_steps_after_reset= env_cfg.ignore_stuck_for_n_steps_after_reset,
        game_speed= env_cfg.game_speed,
        countdown_speed= env_cfg.countdown_speed,
        waitforstep_timeout_in_s= env_cfg.waitforstep_timeout_in_s,
        startposition_accuracy_threshold= env_cfg.startposition_accuracy_threshold,
        gamma= cfg.sb3.algorithm_params.gamma # TODO : This is maybe not the greatest idea, as soon as we work with smth other than sb3 this has to go
    )
    return constructor_kwargs

def get_environment(cfg : TrainConfig, ipcommandsender : IPCommandSender, test : bool = False) -> gym.Env:
    """Initializes environment according to given configuration file and applies wrappers, if specified in conifg.
    Args:
        cfg (TrainConfig)       : Configuration used to initialize environment
        control_queue (Queue)   :  used by environment to send controls to Process-Wrapper
        response_queue (Response) :  Queue used by environment to get responses by ProcessWrapper
        test (bool == False)    : In addition to setting test in config, setting this true always returns a test environment and does not apply any wrappers to the environment.
    Returns:
        TMNF_Single_Agent_Env (gym.Env) : Environment implementing a gym-interface to interact with trackmania
    """

    obs_manager = get_observation_manager_from_cfg(cfg = cfg, wrap_obs_in_test = cfg.rl_env.env.wrap_obs_in_test, normalize=cfg.rl_env.env.normalize_obs)
    reward_calculator = get_reward_calculator_from_cfg(reward_calculator_cfg = cfg.rl_env.reward_manager, normalize=cfg.rl_env.env.normalize_rewards)
    termination_manager = get_termination_manager(termination_cfg= cfg.rl_env.termination_manager)

    constructor_kwargs = _get_env_constructor_args(cfg, ipcommandsender, obs_manager, reward_calculator, termination_manager)
    
    if not cfg.rl_env.env.continuous_actions and not cfg.rl_env.env.test or test:
        # NOTE: I know this is ugly and may lead to confusion but its just for testing purposes we dont plan on
        # adopting this probably
        if cfg.rl_env.env.fake_cont:
            tm_env = FakeContinuousTMNF_Single_Agent_Env(**constructor_kwargs)
        else:
            tm_env = TMNF_Single_Agent_Env(**constructor_kwargs)
    else:
         tm_env = ContinuousTMNF_Single_Agent_Env(**constructor_kwargs, actiondim= cfg.rl_env.env.actiondim)

    if cfg.rl_env.env.test or test:
            constructor_kwargs.update(dict(platform = cfg.platforms.os))
            tm_env = TestEnvironment(**constructor_kwargs)
    
    tm_env.orientationless_respawn_manager = OrientationlessRespawnManager(respawn_coordinates=OrientationlessRespawnManager.get_respawns_for_very_long_checkpoints())

    return tm_env
