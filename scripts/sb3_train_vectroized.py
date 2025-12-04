import sys, os
# TODO : <- i don't want this here and it shouldnt have to be here!!!
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

# Hydra related imports
import hydra
import traceback
from hydra.core.hydra_config import HydraConfig
from typing import Optional


from configs.config import TrainConfig

from trackmania_env.envs.sec_env import CrashProofEnvironment
from utils.hydra_wandb_utils import load_and_merge_platform, secure_attribute_retrieval
from utils.introscreen import introscreen
from trackmania_env.envs.vectorized import VectorizedTMEnvironment, SB3Vectorized

from tmn_sb3.utils.from_cfg import get_model_from_config
from stable_baselines3.common.vec_env import VecNormalize

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "../configs",
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
    introscreen(cfg, askstart=secure_attribute_retrieval(lambda : cfg.ask_start, default=True))
    tracks = ["very_long_checkpoints.Challenge.Gbx", "Level1.Challenge.Gbx"]
    N_ENVS = 1
    tm_env = SB3Vectorized(n_envs = N_ENVS, tracks=tracks, cfg=cfg, obs_as_dict=True)
    try:
        model = get_model_from_config(cfg = cfg, tm_env = tm_env)
        model.learn(**cfg.learn_args)
    except Exception as e:
        traceback.print_exc()

    except KeyboardInterrupt as kinterrupt:
        print("KeyboardInterrupt")

    finally:
        # Finalize training and close game all processes.
        tm_env.close()

if __name__ == "__main__": 
    main()
