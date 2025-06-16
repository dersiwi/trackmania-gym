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

# gymnasium environment wrapper 
from gymnasium import ObservationWrapper
from stable_baselines3.common.monitor import Monitor

# extractor imports
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.base_class import BaseAlgorithm
from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
from trackmania_env.envs.testenv_single_agent import TestEnvironment

from trackmania_env.observations.observations import get_observation_manager
from trackmania_env.observations.observation_manager import ObservationManager
from trackmania_env.observations.observation_test import ObservationTest
 
from trackmania_env.observations.linesight_obs_wrapper import LinesightObservationWrapper

from configs.config import TrainConfig
from utils.printutils import print_model_params
from trackmania_env.utils.init_linesight_obs import get_linesight_obs_instance
from trackmania_env.rewards.getrewards import get_reward_calculator
import glob

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "../configs",
    "config_name": "train.yaml",
}

from utils.hydra_wandb_utils import get_models, init_and_login_wandb

@hydra.main(**_HYDRA_PARAMS)
def main(cfg : TrainConfig):

    HYDRA_RUN_DIR = HydraConfig.get().run.dir
    model_dir = os.path.join(HYDRA_RUN_DIR, "models")
    best_model_path = os.path.join(model_dir, "best_model")
    checkpoint_path = os.path.join(model_dir, "checkpoints")
    os.makedirs(best_model_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    # Start Weights and Biases login
    run, run_id = init_and_login_wandb(cfg, wandbdir=HYDRA_RUN_DIR)
    RUN_ID_IN_HYDRA_LOG_DIR = os.path.join(HYDRA_RUN_DIR, run_id)

    # Instanciate GMI, TMNF-Environment and start TMi-Interaction process.
    tmi_process, control_queue, response_queue = start_process_and_wait_for_startsignal(cfg.platforms, cfg.gmi, cfg.image.width, cfg.image.height)

    try:
        obs_manager = get_observation_manager(cfg)
            
        if cfg.rl_env.env.test:
            TM_ENV_CLASS = TestEnvironment
        else:
            TM_ENV_CLASS = TMNF_Single_Agent_Env

        tm_env = TM_ENV_CLASS(command_queue=control_queue,
                                        response_queue=response_queue, 
                                        obs_manager=obs_manager,
                                        reward_calculator=get_reward_calculator(cfg),
                                        max_steps_before_reset=cfg.rl_env.env.max_steps_until_reset,
                                        game_speed=cfg.rl_env.env.game_speed)
             
        # apply (Observation)-wrappers to the environment
        for _, wrapper_conf in cfg.rl_env.wrappers.items():
            wrapper : ObservationWrapper = hydra.utils.instantiate(wrapper_conf)
            print(f"Wrapping environment in {wrapper.__class__.__name__}")
            tm_env = wrapper(env=tm_env)
        
        # get algorithm and start learning process
        vision_model, model = get_models(cfg, tm_env, print_params = True,run_id=RUN_ID_IN_HYDRA_LOG_DIR)
        # for sb3 the type would be BaseCallBack. For other callbacks we would need to manually write the other types.
        # TODO check if callbacks have the same class they inherit from 
        wb3_callback = hydra.utils.instantiate(cfg.wandb_callbacks)(model_save_path=RUN_ID_IN_HYDRA_LOG_DIR)  if cfg.wandb.use else None  

        # Eval Callback – save best model based on reward
        eval_callback = EvalCallback(
            tm_env,
            best_model_save_path=best_model_path,
            log_path=os.path.join(HYDRA_RUN_DIR, "eval_logs"),
            eval_freq=cfg.wandb.eval_freq,
            deterministic=True,
            render=False,
        )

        # Checkpoint Callback – save model every N steps
        checkpoint_callback = CheckpointCallback(
            save_freq=cfg.wandb.checkpoint_freq,
            save_path=checkpoint_path,
            name_prefix="checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        callback = CallbackList([eval_callback, checkpoint_callback,wb3_callback])
        model.learn(**cfg.learn_args, callback=callback)

        final_model =  os.path.join(HYDRA_RUN_DIR, "model.zip")
        best_model = os.path.join(best_model_path, "best_model.zip")
        checkpoint_files = glob.glob(os.path.join(checkpoint_path, "*.zip"))

        # always save final model
        model.save(os.path.join(HydraConfig.get().run.dir, "model"))

        if cfg.wandb.use:
            # Upload best model
            if os.path.exists(best_model):
                best_artifact = wandb.Artifact("best_model", type="model")
                best_artifact.add_file(best_model)
                run.log_artifact(best_artifact)

            # Upload final model
            final_artifact = wandb.Artifact("final_model", type="model")
            final_artifact.add_file(final_model)
            run.log_artifact(final_artifact)

            # Upload checkpoints
            for ckpt_file in checkpoint_files:
                ckpt_artifact = wandb.Artifact("checkpoint_model", type="model")
                ckpt_artifact.add_file(ckpt_file)
                run.log_artifact(ckpt_artifact)

            run.finish()
        
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