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

from utils.experiment_managers.sb3_exp_manager import Final_Best_CP_Sb3_ExperimentManager

from tmn_sb3.utils.from_cfg import get_model_from_config

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

    tm_env = CrashProofEnvironment(cfg)
    try:
        tm_env.init_environment()
        
        exp_manager = Final_Best_CP_Sb3_ExperimentManager(
            cfg = cfg,
            hydra_run_dir= HydraConfig.get().run.dir,
            resume="must" if run_id else None,
            env= tm_env,
            eval_freq= cfg.wandb.eval_freq,
            checkpoint_freq= cfg.wandb.checkpoint_freq,
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
        tm_env.finalize_process(reinit=False)

if __name__ == "__main__": 
    main()
