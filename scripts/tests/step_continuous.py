import sys, os
# TODO : <- i don't want this here and it shouldnt have to be here!!!
sys.path.append(os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__), '..'), '..'))) # TODO : <- i don't want this here and it shouldnt have to be here!!!

# Hydra related imports
import hydra
import traceback
import random
import numpy as np
from hydra.core.hydra_config import HydraConfig
from typing import Optional


from configs.config import TrainConfig

from trackmania_env.envs.sec_env import CrashProofEnvironment
from utils.hydra_wandb_utils import load_and_merge_platform, secure_attribute_retrieval
from utils.introscreen import introscreen
from trackmania_env.utils.actionmap import ACTION_MAP
from trackmania_env.utils.spacetransform import SpaceTransformer
from tmn_sb3.utils.from_cfg import get_model_from_config

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "../../configs",
    "config_name": "train.yaml",
}

@hydra.main(**_HYDRA_PARAMS)
def main(cfg : TrainConfig, run_id : Optional[str] = None):
    """
    Main Trainings script for training with stable-baslines3.
    Args:
        cfg (TrainConfig)   : Configuration file for current training run (inferred by hydra)
        run_id (str)        : Run id if weights and biases run is to be resumed
    """
    cfg = load_and_merge_platform(cfg)

    N_ENVS = 2
    tracks = ["very_long_checkpoints.Challenge.Gbx", "curvy.Challenge.Gbx", "Level1.Challenge.Gbx"]
    obs_as_dict = False
    tm_env = CrashProofEnvironment(cfg)
    tm_env.init_environment()

    print(tm_env.observation_space)
    print(tm_env.action_space)
        
    try:
        o, info = tm_env.reset()
        for i in range(100000):
            o, rew, term, trun, inf = tm_env.step(np.random.random((2,)) * 2 - 1)
            if i %  100 == 0:
                print(f"Completed {i} steps")




    except Exception as e:
        traceback.print_exc()

    except KeyboardInterrupt as kinterrupt:
        print("KeyboardInterrupt")

    finally:
        # Finalize training and close game all processes.
        for env in tm_env.envs:
            env.finalize_process(reinit=False)

if __name__ == "__main__": 
    main()
