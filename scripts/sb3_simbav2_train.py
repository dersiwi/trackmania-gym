
import sys, os
# TODO : <- i don't want this here and it shouldnt have to be here!!!
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

# Hydra related imports
import hydra
import traceback

from hydra.core.hydra_config import HydraConfig
from typing import Optional


from configs.config import TrainConfig

from trackmania_env.envs.vectorized import SB3Vectorized
from trackmania_env.envs.sec_env import CrashProofEnvironment

from utils.hydra_wandb_utils import load_and_merge_platform, secure_attribute_retrieval
from utils.introscreen import introscreen
from utils.experiment_managers.sb3_exp_manager import Sb3ExperimentManager

from tmn_sb3.utils.from_cfg import get_model_from_config
from multiprocessing import Lock

import omegaconf
import math #this will be used by the omegaconf resolver later

from tmn_sb3.simbav2.normalizers import SimbaVecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from gymnasium.spaces import Dict

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

    def eval_resolver(s: str):
        return eval(s)

    omegaconf.OmegaConf.register_new_resolver("eval", eval_resolver)
    omegaconf.OmegaConf.resolve(cfg)

    assert not cfg.rl_env.env.normalize_obs, "Turn of the obs normalization of the env simbav2 uses its own"
    assert not cfg.rl_env.env.normalize_rewards, "Turn of the reward normalization of the env simbav2 uses its own"
    assert cfg.rl_env.env.continuous_actions, "The current implementation of sac simbav2 works only with continuous actions"


    introscreen(cfg, askstart=secure_attribute_retrieval(lambda : cfg.ask_start, default=True))
    
    if cfg.vectorized.vectorize:
        tm_env = SB3Vectorized(n_envs = cfg.vectorized.n_envs, 
                               tracks=cfg.vectorized.tracks, 
                               cfg=cfg, obs_as_dict=True, step_parallel=True, 
                               lock = Lock())
    else:
        def make_env():
            env = CrashProofEnvironment(cfg)
            env.init_environment()
            env = Monitor(env)
            return env
        
        tm_env = DummyVecEnv([make_env])
        # the SimbaVecNormalize chnages every dtype to float32 -> this leads to the image extractor not checking that the image obs space is an image
        obs_keys = None
        if isinstance(tm_env.observation_space,Dict):
            obs_keys = tm_env.observation_space.spaces.keys()
        tm_env = SimbaVecNormalize(tm_env,g_max=cfg.sb3.algorithm_params.max_v,norm_obs_keys=obs_keys)
    
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
            for env in tm_env.venv.envs:
                env.finalize_process(reinit=False)
        

if __name__ == "__main__": 
    main()
