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
from game_interaction.ipc_fields import IPCCommands

from configs.config import TrainConfig
from trackmania_env.envs.enivonrments import get_environment
import glob
from tqdm import tqdm

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "../configs",
    "config_name": "train.yaml",
}

from utils.hydra_wandb_utils import get_models, init_and_login_wandb, BeforeAndAfterTraining
import queue  # Add this if you're using standard Queue
import traceback
from queue import Empty  # Catch this specific exception
# If using multiprocessing.Queue, you may need:
# from multiprocessing.queues import Empty


def main(cfg: TrainConfig, speed):
    queue_empty_error = []
    cfg.rl_env.env.game_speed = speed
    cfg.rl_env.env.countdown_speed = speed
    cfg.gmi.track = "straight_line.Challenge.Gbx"
    cfg.gmi.reference_line = "tracks/reference_line/straight_line.npy"
    # Start environment and interaction process
    tmi_process, control_queue, response_queue = start_process_and_wait_for_startsignal(
        cfg, cfg.rl_env.obs_manager.img_width, cfg.rl_env.obs_manager.img_height)

    try:
        tm_env = get_environment(cfg, control_queue, response_queue)
        tm_env.reset()
        for i in tqdm(range(8000000), desc=f"Speed {speed}"):
            try:
                observations, reward, terminated, truncated, info  = tm_env.step(0)  # just drive straight
                queue_empty_error.append(0)  # Step successful
                if terminated or truncated: tm_env.reset()
            except Empty:
                print(f"[Warning] Queue.Empty at iteration {i}")
                queue_empty_error.append(1)  # Queue empty error
                tm_env.reset()
                continue
            except Exception as step_e:
                print(f"[Error] Unknown exception at step {i}")
                traceback.print_exc()
                queue_empty_error.append(-1)
                tm_env.reset()  # Other unknown error
                continue

    except Exception as e:
        traceback.print_exc()

    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    finally:
        control_queue.put(IPCCommands.get_end_syncloop_command(1000))
        tmi_process.join()

        # Optionally: Save queue_empty_error array for diagnostics
        with open(f"queue_empty_log_{speed}.txt", "w") as f:
            f.write(str(queue_empty_error))


def run_for_speed(speed):
    from omegaconf import OmegaConf
    # Use Hydra's compose to manually construct the config
    from hydra import initialize, compose

    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(config_name="train.yaml")
        main(cfg, speed)


if __name__ == "__main__":
    speeds = [60, 70, 80]
    for s in speeds:
        run_for_speed(s)