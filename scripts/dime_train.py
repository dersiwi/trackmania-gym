import sys
import os

# 1. Path setup for DIME and project root
script_dir = os.path.dirname(os.path.abspath(__file__))
dime_root = os.path.abspath(os.path.join(script_dir, '..', 'DIME'))

if dime_root not in sys.path:
    sys.path.insert(0, dime_root)

project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import jax
import time
import hydra
import wandb
import omegaconf
import traceback
import gymnasium as gym

from common.buffers import DMCCompatibleDictReplayBuffer
from diffusion.dime import DIME as dime
from omegaconf import DictConfig, OmegaConf
from models.utils import is_slurm_job
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList
from models.actor_critic_evaluation_callback import EvalCallback

from trackmania_env.envs.sec_env import CrashProofEnvironment
from utils.hydra_wandb_utils import load_and_merge_platform

def _create_alg(full_cfg: DictConfig, dime_cfg: DictConfig):
    try:
        import myosuite
    except ImportError:
        print("myosuite not installed")
        pass

    training_env = CrashProofEnvironment(full_cfg)
    training_env.init_environment()

    tensorboard_log_dir = f"./logs/{dime_cfg.wandb['group']}/{dime_cfg.wandb['job_type']}/seed={str(dime_cfg.seed)}/"
    eval_log_dir = f"./eval_logs/{dime_cfg.wandb['group']}/{dime_cfg.wandb['job_type']}/seed={str(dime_cfg.seed)}/eval/"

    # 3. Pass ONLY the instantiated dime_cfg to the model
    model = dime(
        "MultiInputPolicy" if isinstance(training_env.observation_space, gym.spaces.Dict) else "MlpPolicy",
        env=training_env,
        model_save_path=None,
        save_every_n_steps=int(dime_cfg.tot_time_steps / 100000),
        cfg=dime_cfg, 
        tensorboard_log=tensorboard_log_dir,
        replay_buffer_class=None
    )

    os.makedirs(eval_log_dir, exist_ok=True)

    # 4. Use dime_cfg for callbacks as well
    eval_callback = EvalCallback(
        training_env,
        jax_random_key_for_seeds=dime_cfg.seed,
        best_model_save_path=None,
        log_path=eval_log_dir,
        eval_freq=max(300000 // dime_cfg.log_freq, 1),
        n_eval_episodes=5, 
        deterministic=True, 
        render=False
    )

    if dime_cfg.wandb["activate"]:
        callback_list = CallbackList([eval_callback, WandbCallback(verbose=0)])
    else:
        callback_list = CallbackList([eval_callback])
    
    return model, callback_list

def initialize_and_run(cfg):

    temp_cfg = cfg.copy()
    # Remove rl_env from the configuration
    with omegaconf.open_dict(temp_cfg):
        if "rl_env" in temp_cfg:
            del temp_cfg["rl_env"]
            print("Removed rl_env from config before DIME instantiation")

    dime_cfg = hydra.utils.instantiate(temp_cfg)
    seed = dime_cfg.seed
    
    if dime_cfg.wandb["activate"]:
        name = f"seed_{seed}"
        # We save the full original config to wandb for reproducibility
        wandb_config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(
            settings=wandb.Settings(_service_wait=300),
            project=dime_cfg.wandb["project"],
            group=dime_cfg.wandb["group"],
            job_type=dime_cfg.wandb["job_type"],
            # name=name,
            config=wandb_config,
            entity=dime_cfg.wandb["entity"],
            sync_tensorboard=True,
        )
        if is_slurm_job():
            print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}")
            wandb.summary['SLURM_JOB_ID'] = os.environ.get('SLURM_JOB_ID')

    model, callback_list = _create_alg(cfg,dime_cfg)       
    model.learn(total_timesteps=dime_cfg.tot_time_steps, progress_bar=True, callback=callback_list)

@hydra.main(version_base=None, config_path="../configs", config_name="dime_train")
def main(cfg: DictConfig) -> None:
    cfg = load_and_merge_platform(cfg)
    try:
        starting_time = time.time()
        # Use cfg.dime to check flags since that structure exists in the raw config
        if cfg.use_jit:
            initialize_and_run(cfg)
        else:
            with jax.disable_jit():
                initialize_and_run(cfg)
                
        end_time = time.time()
        print(f"Training took: {(end_time - starting_time)/3600} hours")
        
    except Exception as ex:
        print("-- exception occured. traceback :")
        traceback.print_tb(ex.__traceback__)
        print(ex, flush=True)
        print("--------------------------------\n")
        traceback.print_exception(ex)
    
    finally:
        # Safe check for wandb finish
        if cfg.wandb["activate"]:
            wandb.finish()

if __name__ == "__main__":
    main()
