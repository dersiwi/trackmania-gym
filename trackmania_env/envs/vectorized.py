import torch
import numpy as np
import gymnasium as gym


from trackmania_env.envs.sec_env import CrashProofEnvironment
from configs.config import TrainConfig


class VectorizedTMEnvironment(gym.Env):

    def __init__(self, n_envs : int, tracks : list[str], cfg : TrainConfig):
        self.n_envs = n_envs
        self.tracks = tracks
        self.cfg = cfg
        
        port = self.cfg.gmi.port   
        self.envs : list[CrashProofEnvironment] = [CrashProofEnvironment(train_cfg=self.cfg, port = port+i) for i in range(self.n_envs)]
        for i in range(self.n_envs):
            self.envs.append() 
            self.envs[i].init_environment()
            self.obs_space = self.envs[i].observation_space


    def step(self, action : torch.Tensor | np.ndarray):
        assert action.shape[0] == self.n_envs
        info = []
        rewards = np.zeros((self.n_envs, ))
        terminated = np.zeros((self.n_envs, ))
        truncated = np.zeros((self.n_envs, ))

        envobs = []
 
        for i in range(self.n_envs):
            obs, rew, term, trun, envinfo = self.envs[i].step(action[i])
            info.append(envinfo)
            terminated[i] = term
            truncated[i] = trun
            rewards[i] = rew
            envobs.append(obs)            
        
        return self._batch_observations(envobs), rewards, terminated, truncated, info
            
    def _batch_observations(self, envobs : list) -> any:
        if self.obs_space == gym.spaces.Dict:
            observations = {}
            for term in envobs[0]:
                batched_obs = np.zeros_like(envobs[0][term])
                batched_obs = batched_obs[np.newaxis, :]
                observations[term] = batched_obs
        else:
            observations = np.zeros((self.n_envs, ))


        for i, envo in enumerate(envobs):
            if self.obs_space == gym.spaces.Dict:
                for term in envo:
                    observations[term][i] = envo[term]
            else:
                observations[i] = envo
        return observations
    
    
    def reset(self, *, seed = None, options = None):
        for i in range(self.n_envs):
            observation, info = self.envs[i].reset()