# utils
import sys, os
# TODO : <- i don't want this here and it shouldnt have to be here!!!
sys.path.append(os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__), '..'), '..'))) # TODO : <- i don't want this here and it shouldnt have to be here!!!

# Hydra related imports
import hydra
import omegaconf
from hydra.core.hydra_config import HydraConfig
import traceback

# Weights and Biases related imports
import wandb

# imports for communication between TMInterface and environment
from game_interaction.run_multiprocess_wrapper import start_process_and_wait_for_startsignal
from game_interaction.ipc_fields import IPCCommands

from configs.config import TrainConfig
from trackmania_env.envs.enivonrments import get_environment
import glob
import random
import time

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "../../configs",
    "config_name": "train.yaml",
}

from utils.hydra_wandb_utils import get_models, init_and_login_wandb, BeforeAndAfterTraining
def print_progress_bar(iteration, total, length=40, fill='█', suffix=''):
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\rProgress |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()  # Move to next line

@hydra.main(**_HYDRA_PARAMS)
def main(cfg : TrainConfig):


    # Instanciate GMI, TMNF-Environment and start TMi-Interaction process.
    tmi_process, control_queue, response_queue = start_process_and_wait_for_startsignal(cfg, cfg.image.width, cfg.image.height)

    try:
        tm_env = get_environment(cfg, control_queue, response_queue)
        obs, info = tm_env.reset()
        n_steps = 1000000
        for i in range(n_steps):
            processed_obs, reward, terminated, truncated, info = tm_env.step(random.randint(0, 11))
            if i % 4098 == 0 and i > 0:
                time.sleep(15)
            if terminated or truncated:
                tm_env.reset()
            print_progress_bar(i, n_steps, suffix=f"[{i}/{n_steps}]")
        
    except Exception as e:
        traceback.print_exc()

    except KeyboardInterrupt as kinterrupt:
        print("KeyboardInterrupt")

    finally:
        # Finalize training and close game all processes.
        control_queue.put(IPCCommands.get_end_syncloop_command(1000)) #1000 doesnt matter.
        tmi_process.join()


if __name__ == "__main__": 
    main()