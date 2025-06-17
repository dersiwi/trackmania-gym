import torch as th
import numpy as np
from stable_baselines3.common.buffers import DictReplayBuffer


class CudaDictReplayBuffer(DictReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device="cuda", n_envs=1):
        super().__init__(buffer_size, observation_space, action_space, device=device, n_envs=n_envs)

        # Allocate dict-based torch buffers on GPU
        self.observations = {
            key: th.zeros((self.buffer_size, *space.shape), dtype=th.float32, device=device)
            for key, space in observation_space.spaces.items()
        }
        self.next_observations = {
            key: th.zeros((self.buffer_size, *space.shape), dtype=th.float32, device=device)
            for key, space in observation_space.spaces.items()
        }

        # Override action, reward, done buffers to reside on GPU
        self.actions = th.zeros((self.buffer_size,) + action_space.shape, dtype=th.float32, device=device)
        self.rewards = th.zeros((self.buffer_size,), dtype=th.float32, device=device)
        self.dones = th.zeros((self.buffer_size,), dtype=th.bool, device=device)

    def add(self, obs, next_obs, action, reward, done, infos):
        i = self.pos

        # Store obs and next_obs
        for key in self.observations:
            self.observations[key][i] = th.as_tensor(obs[key], dtype=th.float32, device=self.device)
            self.next_observations[key][i] = th.as_tensor(next_obs[key], dtype=th.float32, device=self.device)

        # Store action, reward, done
        self.actions[i] = th.as_tensor(action, dtype=th.float32, device=self.device)
        self.rewards[i] = th.tensor(reward, dtype=th.float32, device=self.device)
        self.dones[i] = th.tensor(done, dtype=th.bool, device=self.device)

        # Update buffer position
        self.pos = (self.pos + 1) % self.buffer_size
        self.full = self.full or self.pos == 0

    def sample(self, batch_size, env=None):
        idx = np.random.randint(0, self.buffer_size if self.full else self.pos, size=batch_size)

        obs_batch = {key: buf[idx] for key, buf in self.observations.items()}
        next_obs_batch = {key: buf[idx] for key, buf in self.next_observations.items()}

        return (
            obs_batch,
            self.actions[idx],
            next_obs_batch,
            self.dones[idx],
            self.rewards[idx],
        )
