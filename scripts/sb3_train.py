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
from utils.hydra_wandb_utils import get_models, init_and_login_wandb, BeforeAndAfterTraining, load_and_merge_platform


_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "../configs",
    "config_name": "train.yaml",
}

@hydra.main(**_HYDRA_PARAMS)
def main(cfg : TrainConfig, run_id : Optional[str] = None):
    """Main Trainings script for training with stable-baslines3.
    Args:
        cfg (TrainConfig)   : Configuration file for current training run (inferred by hydra)
        run_id (str)        : Run id if weights and biases run is to be resumed
    """
    cfg = load_and_merge_platform(cfg)

    hydra_run_dir = HydraConfig.get().run.dir
    resume = "must" if run_id else None

    baaf = BeforeAndAfterTraining(hydra_run_dir=HydraConfig.get().run.dir, cfg = cfg, resume=resume)
    baaf.before_training()

    tm_env = CrashProofEnvironment(cfg)
    try:
        tm_env.init_environment()
        
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
        tm_env.finalize_process(reinit=False)

if __name__ == "__main__": 
    main()