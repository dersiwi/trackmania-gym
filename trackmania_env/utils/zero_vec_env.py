import gym
from gym import spaces
import numpy as np

class ZeroVecEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, n_envs=8, obs_dim=4, action_dim=2):
        super().__init__()
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.observation_space = spaces.Box(
            low=0.0,
            high=0.0,
            shape=(n_envs, obs_dim),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(n_envs, action_dim),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        obs = np.zeros((self.n_envs, self.obs_dim), dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        obs = np.zeros((self.n_envs, self.obs_dim), dtype=np.float32)
        reward = np.zeros(self.n_envs, dtype=np.float32)
        terminated = np.zeros(self.n_envs, dtype=bool)
        truncated = np.zeros(self.n_envs, dtype=bool)
        info = {}

        return obs, reward, terminated, truncated, info
