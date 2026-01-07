import sys, os
# TODO : <- i don't want this here and it shouldnt have to be here!!!
sys.path.append(os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__), '..'), '..'))) # TODO : <- i don't want this here and it shouldnt have to be here!!!

# Hydra related imports
import hydra
import traceback
import numpy as np
from typing import Optional


from configs.config import TrainConfig

from trackmania_env.envs.sec_env import CrashProofEnvironment
from utils.hydra_wandb_utils import load_and_merge_platform
from trackmania_env.utils.actionmap import ActionMode

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

    tm_env = CrashProofEnvironment(cfg)
    tm_env.init_environment()

    print(tm_env.observation_space)
    print(tm_env.action_space)

    speed, incspeed = 0, False
        
    try:
        o, info = tm_env.reset()
        for i in range(100000):
            action = ActionMode.generate_random_action(mode=ActionMode.CONTINUOUS_2D) if not incspeed else np.array([0.0, speed])
            print(f"[Aciton info]     :      steering : {action[0]},      acceleration {action[1]}")
            o, rew, term, trun, inf = tm_env.step(action)
            
            if incspeed:
                if i % 50 == 0:
                    speed += 0.2
                    print(f"Current speed {speed}")
                if speed > 1:
                    speed = 0

            if i %  100 == 0:
                print(f"Completed {i} steps")
            if term or trun:
                tm_env.reset()




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
