import torch
import numpy as np
import gymnasium as gym


from trackmania_env.envs.sec_env import CrashProofEnvironment
from trackmania_env.utils.spacetransform import SpaceTransformer
from configs.config import TrainConfig


class VectorizedTMEnvironment(gym.Env):

    def __init__(self, n_envs : int, tracks : list[str], cfg : TrainConfig):
        self.n_envs = n_envs
        self.tracks = tracks
        self.cfg = cfg
        
        port = self.cfg.gmi.port   
        self.envs : list[CrashProofEnvironment] = [CrashProofEnvironment(train_cfg=self.cfg, port = port+i, return_obs_as_dict = False) for i in range(self.n_envs)]
        for i in range(self.n_envs):
            self.envs.append() 
            self.envs[i].init_environment()
        
        self.transformer = SpaceTransformer.get_instance()
        self.transformer.expect_vectorized(self.n_envs)
        self.obs_size = self.transformer.expected_dim


    def step(self, action : torch.Tensor | np.ndarray):
        assert action.shape[0] == self.n_envs
        info = []
        observations = np.zeros((self.n_envs, self.obs_size))
        rewards = np.zeros((self.n_envs, ))
        terminated = np.zeros((self.n_envs, ))
        truncated = np.zeros((self.n_envs, ))

        for i in range(self.n_envs):
            obs, rew, term, trun, envinfo = self.envs[i].step(action[i])
            observations[i] = obs
            terminated[i] = term
            truncated[i] = trun
            rewards[i] = rew
            info.append(envinfo)
        
        return observations, rewards, terminated, truncated, info
            
    def reset(self, *, seed = None, options = None):
        for i in range(self.n_envs):
            observation, info = self.envs[i].reset()