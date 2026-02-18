from bro_torch.replay_buffer import ReplayBuffer
import numpy as np
import torch

class VecReplayBuffer(ReplayBuffer):
    def __init__(self, n_envs, buffer_size, observation_size, action_size, device = 'cpu'):
        super().__init__(buffer_size, observation_size, action_size, device)
        self.num_envs = n_envs

    def add(self, observations: np.ndarray, next_observations: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray):
        """
        observations: (num_envs, observation_size)
        actions: (num_envs, action_size)
        rewards/dones: (num_envs,)
        """
        end_idx = self.insert_index + self.num_envs
        
        if end_idx <= self.buffer_size:
            self.observations[self.insert_index:end_idx] = observations
            self.actions[self.insert_index:end_idx] = actions
            self.rewards[self.insert_index:end_idx] = rewards
            self.dones[self.insert_index:end_idx] = dones
            self.next_observations[self.insert_index:end_idx] = next_observations
        else:
            overflow = end_idx - self.buffer_size
            keep = self.num_envs - overflow
            
            self.observations[self.insert_index:] = observations[:keep]
            self.actions[self.insert_index:] = actions[:keep]
            self.rewards[self.insert_index:] = rewards[:keep]
            self.dones[self.insert_index:] = dones[:keep]
            self.next_observations[self.insert_index:] = next_observations[:keep]
            
            self.observations[:overflow] = observations[keep:]
            self.actions[:overflow] = actions[keep:]
            self.rewards[:overflow] = rewards[keep:]
            self.dones[:overflow] = dones[keep:]
            self.next_observations[:overflow] = next_observations[keep:]

        # Update pointers
        self.insert_index = (self.insert_index + self.num_envs) % self.buffer_size
        self.size = min(self.size + self.num_envs, self.buffer_size)

    def to_tensor(self, array: np.ndarray):
        return torch.from_numpy(array).to(self.device)