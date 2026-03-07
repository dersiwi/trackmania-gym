from typing import Optional 
from configs.config import TrainConfig
from multiprocessing import Lock
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from trackmania_env.envs.sec_env import CrashProofEnvironment
from trackmania_env.envs.vectorized import *


def make_env(cfg : TrainConfig) -> Optional[SB3Vectorized | CrashProofEnvironment]:

    """Have to call tm_env.init_environment() after you get it."""
    if cfg.vectorized.vectorize:
        tm_env = SB3Vectorized(n_envs = cfg.vectorized.n_envs, 
                               tracks=cfg.vectorized.tracks, 
                               cfg=cfg, obs_as_dict=cfg.vectorized.obs_as_dict, step_parallel=cfg.vectorized.step_parallel, 
                               lock = Lock())
        if cfg.vectorized.normalize_per_batch:
            assert not cfg.rl_env.env.normalize_rewards, "You have activated normnalized-rewards per environment and batch-normalization on the vectorized environment. \
                Either disable per-batch normalization (not recommended when trainng vectorized) or disable per-environment reward normalization."
            assert not cfg.rl_env.env.normalize_obs, "You have activated normalize-observations per environment and batch-normalization on the vectorized environment. \
                Either disable per-batch normalization (not recommended when trainng vectorized) or disable per-environment obs normalization."
            tm_env = VecNormalize(tm_env, training = cfg.vectorized.training)
    else:
        tm_env = CrashProofEnvironment(cfg)
    return tm_env