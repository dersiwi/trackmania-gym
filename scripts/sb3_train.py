# utils
import sys, os
# TODO : <- i don't want this here and it shouldnt have to be here!!!
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

import time
from typing import Optional

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

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "../configs",
    "config_name": "train.yaml",
}

from utils.hydra_wandb_utils import get_models, init_and_login_wandb, BeforeAndAfterTraining

def train_agent(
    cfg : TrainConfig,
    model_save_path: Optional[str] = None,
    replay_buffer_save_path: Optional[str] = None,
    run_id : Optional[str] = None
):
    """
    Sets up and runs a reinforcement learning training process.

    Args:
        cfg: The configuration object from Hydra.
        model_save_path (Optional[str]): Path to a pre-trained model file to load.
        replay_buffer_save_path (Optional[str]): Path to a replay buffer file to load.
        run_id (Optional[str]): The unique ID for the WandB run, used to resume a crashed or interrupted training session.
    """
    hydra_run_dir=HydraConfig.get().run.dir
    resume = "must" if run_id else None

    baaf = BeforeAndAfterTraining(hydra_run_dir=hydra_run_dir, cfg = cfg,run_id = run_id, resume = resume )
    baaf.before_training()

    # Instanciate GMI, TMNF-Environment and start TMi-Interaction process.
    tmi_process, control_queue, response_queue = start_process_and_wait_for_startsignal(cfg, cfg.rl_env.obs_manager.img_width, cfg.rl_env.obs_manager.img_height)

    try:
        tm_env = get_environment(cfg, control_queue, response_queue)
        
        # get algorithm and start learning process
        vision_model, model = get_models(
            cfg,
            tm_env,
            print_params = True,
            run_id = baaf.get_tensorboard_login_identifier(),
            load_model_path = model_save_path,
            load_replay_buffer_path = replay_buffer_save_path
            )

        model.learn(**cfg.learn_args, callback=baaf.get_callbacks_for_training(tm_env))

        baaf.after_training(model)
        

    except KeyboardInterrupt as kinterrupt:
        print("KeyboardInterrupt")

    # most likely Queue-Empty error
    except RuntimeError as run_err:
        run_id = baaf.run_id
        steps_trained = model.num_timesteps

        savepoint_dir = os.path.join(hydra_run_dir, "savepoint")
        os.makedirs(savepoint_dir, exist_ok=True)

        model_save_path = model_save_path or os.path.join(savepoint_dir, "model.zip")
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")

         # Save the replay buffer if it exists
        if hasattr(model, 'save_replay_buffer'):
            replay_buffer_save_path = replay_buffer_save_path or os.path.join(savepoint_dir, "replay_buffer.pkl")
            model.save_replay_buffer(replay_buffer_save_path)
            print(f"Replay buffer saved to {replay_buffer_save_path}")

        return run_id, steps_trained, model_save_path, replay_buffer_save_path
    
    except Exception as e:
        traceback.print_exc()

    finally:
        # Finalize training and close game all processes.
        control_queue.put(IPCCommands.get_end_syncloop_command(1000)) #1000 doesnt matter.
        tmi_process.join()
        time.sleep(5)

@hydra.main(**_HYDRA_PARAMS)
def main(cfg : TrainConfig):
    # store the state for resuming
    resume_state = None

    while True:
        # The first time, all paths are None. On subsequent calls, we pass the saved paths.
        if resume_state is None:
            # First run, start from scratch
            result = train_agent(cfg=cfg)
        else:
            # Resume training from the saved state
            run_id, steps_trained, model_path, replay_buffer_path = resume_state

            if steps_trained >= cfg.learn_args.total_timesteps:
                print("Training has already reached the total_timesteps. Breaking loop.")
                break
    
            cfg.learn_args.total_timesteps = cfg.learn_args.total_timesteps - steps_trained
            cfg.learn_args.progress_bar = False # when not set to false another progess bar tries to get created eventhough only one can exist
            result = train_agent(
                cfg=cfg,
                model_save_path=model_path,
                replay_buffer_save_path=replay_buffer_path,
                run_id= run_id
            )

        # If a tuple is returned it means a RuntimeError occurred and we should resume.
        if isinstance(result, tuple):
            print("RuntimeError occurred. Resuming training...")
            resume_state = result
        else:
            # If None or any other value is returned the training is complete or was stopped.
            print("Training finished or was manually stopped.")
            break

if __name__ == "__main__": 
    main()