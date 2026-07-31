import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "trackmania_gym"
for candidate in [REPO_ROOT, SRC_ROOT]:
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.append(candidate_str)

import hydra
import traceback

from hydra.core.hydra_config import HydraConfig
from typing import Optional

from configs.config import TrainConfig

from trackmania_gym.trackmania_env.envs.vectorized import SB3Vectorized
from trackmania_gym.trackmania_env.envs.sec_env import CrashProofEnvironment

from trackmania_gym.utils.hydra_wandb_utils import load_and_merge_platform, secure_attribute_retrieval
from trackmania_gym.utils.introscreen import introscreen
from trackmania_gym.utils.experiment_managers.sb3_exp_manager import Sb3ExperimentManager

from trackmania_gym.tmn_sb3.utils.from_cfg import get_model_from_config
from multiprocessing import Lock


_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(REPO_ROOT / "configs"),
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
    
    if cfg.vectorized.vectorize:
        tm_env = SB3Vectorized(n_envs = cfg.vectorized.n_envs, 
                               tracks=cfg.vectorized.tracks, 
                               cfg=cfg, obs_as_dict=True, step_parallel=True, 
                               lock = Lock())
    else:
        tm_env = CrashProofEnvironment(cfg)
        tm_env.init_environment()
    
    try:
        exp_manager = Sb3ExperimentManager(
            cfg = cfg,
            hydra_run_dir= HydraConfig.get().run.dir,
            resume="must" if run_id else None,
            env= tm_env,
            eval_freq= cfg.wandb.eval_freq,
            checkpoint_freq= cfg.wandb.checkpoint_freq,
            n_envs = cfg.vectorized.n_envs if cfg.vectorized.vectorize else 1
            )

        model = get_model_from_config(cfg = cfg, tm_env = tm_env, print_params= True, run_id= exp_manager.get_tensorboard_login_identifier())
        model.learn(**cfg.learn_args, callback= exp_manager.get_callbacks())
    except Exception as e:
        traceback.print_exc()

    except KeyboardInterrupt as kinterrupt:
        print("KeyboardInterrupt")

    finally:
        exp_manager.after_training(model= model)
        # Finalize training and close game all processes.
        if cfg.vectorized.vectorize:
            tm_env.close()
        else:
            tm_env.finalize_process(reinit=False)
        

if __name__ == "__main__": 
    main()
