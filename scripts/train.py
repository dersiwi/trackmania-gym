# utils
import sys, os
# TODO : <- i don't want this here and it shouldnt have to be here!!!
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

# Hydra related imports
import hydra
import omegaconf
from hydra.core.hydra_config import HydraConfig
import traceback

# Weights and Biases related imports
import wandb

# imports for communication between TMInterface and environment
from game_interaction.run_multiprocess_wrapper import start_process_and_wait_for_startsignal
from game_interaction.process_wrapper import TMIProcessWrapper

# gymnasium environment wrapper 
from gymnasium import ObservationWrapper
from stable_baselines3.common.monitor import Monitor

# extractor imports
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.base_class import BaseAlgorithm
from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
from trackmania_env.envs.testenv_single_agent import TestEnvironment

from trackmania_env.utils.reference_line_manager import ReferenceLineManager
from trackmania_env.observations.observations import get_observation_manager
from trackmania_env.rewards.getrewards import get_reward_calculator
from trackmania_env.terminations.get_termination_manager import get_termination_manager
from trackmania_env.rewards.reward_calculation import RewardLogCallback, AccumRewardLogCallback
from trackmania_env.utils.orientationless_random_respawn_manager import OrientationlessRespawnManager
from configs.config import TrainConfig

import glob

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "../configs",
    "config_name": "train.yaml",
}

from utils.hydra_wandb_utils import get_models, init_and_login_wandb, BeforeAndAfterTraining

@hydra.main(**_HYDRA_PARAMS)
def main(cfg : TrainConfig):

    baaf = BeforeAndAfterTraining(hydra_run_dir=HydraConfig.get().run.dir, cfg = cfg)
    baaf.before_training()

    # Instanciate GMI, TMNF-Environment and start TMi-Interaction process.
    tmi_process, control_queue, response_queue = start_process_and_wait_for_startsignal(cfg.platforms, cfg.gmi, cfg.image.width, cfg.image.height)

    try:
        obs_manager = get_observation_manager(cfg)
            
        if cfg.rl_env.env.test:
            TM_ENV_CLASS = TestEnvironment
        else:
            TM_ENV_CLASS = TMNF_Single_Agent_Env

        tm_env = TM_ENV_CLASS(command_queue=control_queue,
                                        response_queue=response_queue, 
                                        obs_manager=obs_manager,
                                        reward_calculator=get_reward_calculator(cfg),
                                        termination_manger=get_termination_manager(cfg),
                                        reference_line = ReferenceLineManager(cfg.gmi.reference_line),
                                        env_cfg=cfg.rl_env.env)
        tm_env.orientationless_respawn_manager = OrientationlessRespawnManager(respawn_coordinates=OrientationlessRespawnManager.get_respawns_for_very_long_checkpoints())
             
        # apply (Observation)-wrappers to the environment
        for _, wrapper_conf in cfg.rl_env.wrappers.items():
            wrapper : ObservationWrapper = hydra.utils.instantiate(wrapper_conf)
            print(f"Wrapping environment in {wrapper.__class__.__name__}")
            tm_env = wrapper(env=tm_env)
        
        # get algorithm and start learning process
        vision_model, model = get_models(cfg, tm_env, print_params = True, run_id=baaf.get_tensorboard_login_identifier())

        model.learn(**cfg.learn_args, callback=baaf.get_callbacks_for_training(tm_env))

        baaf.after_training(model)
        
    except Exception as e:
        traceback.print_exc()

    except KeyboardInterrupt as kinterrupt:
        print("KeyboardInterrupt")

    finally:
        # Finalize training and close game all processes.
        control_queue.put(TMIProcessWrapper.IPCCommands.get_end_syncloop_command(1000)) #1000 doesnt matter.
        tmi_process.join()


if __name__ == "__main__": 
    main()