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

from trackmania_env.observations.observations import get_observation_manager
from trackmania_env.observations.observation_manager import ObservationManager
from trackmania_env.observations.observation_test import ObservationTest
 
from trackmania_env.observations.linesight_obs_wrapper import LinesightObservationWrapper

from configs.config import TrainConfig
from utils.printutils import print_model_params
from trackmania_env.utils.init_linesight_obs import get_linesight_obs_instance
from trackmania_env.rewards.getrewards import get_reward_calculator
from trackmania_env.rewards.reward_calculation import RewardLogCallback

import glob
import matplotlib.pyplot as plt

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "../configs",
    "config_name": "train.yaml",
}

from utils.hydra_wandb_utils import get_models, init_and_login_wandb

@hydra.main(**_HYDRA_PARAMS)
def main(cfg : TrainConfig):

    HYDRA_RUN_DIR = HydraConfig.get().run.dir

    # Start Weights and Biases login
    run, run_id = init_and_login_wandb(cfg, wandbdir=HYDRA_RUN_DIR)
    RUN_ID_IN_HYDRA_LOG_DIR = os.path.join(HYDRA_RUN_DIR, run_id)

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
                                        max_steps_before_reset=cfg.rl_env.env.max_steps_until_reset,
                                        game_speed=cfg.rl_env.env.game_speed)
             
        test_velocity(tm_env)
        
    except Exception as e:
        traceback.print_exc()

    except KeyboardInterrupt as kinterrupt:
        print("KeyboardInterrupt")

    finally:
        # Finalize training and close game all processes.
        control_queue.put(TMIProcessWrapper.IPCCommands.get_end_syncloop_command(1000)) #1000 doesnt matter.
        tmi_process.join()


def test_velocity(env):
    """
    This test is made to be run onf the straight_line.Challange.GBX map
    """
    terminated = False
    observations, info = env.reset()

    # Velocity components over time
    v_x, v_y, v_z = [], [], []

    # Step counter
    n = 1
    while not terminated:
        # Choose action
        action = 0  # default: go forward
        if n % 50 == 0:
            action = 1  # go left
        # elif n % 13 == 0:
        #     action = 2  # go right

        # Step the environment
        observations, reward, terminated, truncated, info = env.step(action)
        velocity = info["velocity"]
        v_x.append(velocity[0])
        v_y.append(velocity[1])
        v_z.append(velocity[2])

        n += 1

    time = range(len(v_x))  

    plt.figure(figsize=(12, 6))

    plt.subplot(3, 1, 1)
    plt.plot(time, v_x, label='v_x')
    plt.ylabel('v_x')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time, v_y, label='v_y', color='orange')
    plt.ylabel('v_y')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time, v_z, label='v_z', color='green')
    plt.ylabel('v_z')
    plt.xlabel('Time Step')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__": 
    main()

