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

from configs.config import TrainConfig
from trackmania_env.envs.enivonrments import get_environment
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
    tmi_process, control_queue, response_queue = start_process_and_wait_for_startsignal(cfg, cfg.image.width, cfg.image.height)

    try:
        tm_env = get_environment(cfg, control_queue, response_queue)
        
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