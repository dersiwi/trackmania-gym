import numpy as np
import torch as th
from typing import Any, Dict, List, Optional, Union, Type
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env import VecNormalize


class SimbaV2ReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        gamma: float = 0.99,
        g_max: float = 10.0,
        epsilon: float = 1e-8,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination
        )

        # Observation Normalisation section 3.2 from simbav2 paper
        self.obs_rms = RunningMeanStd(shape=self.obs_shape)

        # Reward Normalisation section 4.3 paragraph Reward Bounding and Scaling
        self.returns = np.zeros(n_envs)
        self.return_rms = RunningMeanStd(shape=(1,))
        self.g_r_max = 0.0  # Running max of absolute returns
        self.gamma = gamma
        self.g_max = g_max

        self.epsilon = epsilon

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # update the running obs & reward normalizers
        self.obs_rms.update(obs)

        # G_t = gamma * (1 - done) * G_{t-1} + reward
        self.returns = self.gamma * (1 - done) * self.returns + reward
        self.return_rms.update(self.returns)
        self.g_r_max = max(self.g_r_max, np.max(np.abs(self.returns)))

        return super().add(obs, next_obs, action, reward, done, infos)
    
    # NOTE: This implementation is kind of ugly. SB3 already provides the normalizition methods
    # which call the underling normalize functions of the env. We are currently integration the normalizers
    # themselfs into the Repay buffer. We should really think about doing the other more cleaner way and look
    # if it makes a difference since currently i am also unshure if the would be equivalent in the way the work
    def normalize_obs(
        self,
        obs: Union[np.ndarray, dict[str, np.ndarray]],
        env: Optional[VecNormalize] = None,
    ) -> Union[np.ndarray, dict[str, np.ndarray]]:
         return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)

    def normalize_reward(self, reward: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        var_denominator = np.sqrt(self.return_rms.var + self.epsilon)
        min_required_denominator = self.g_r_max / self.g_max
        denominator = max(var_denominator, min_required_denominator)

        return reward/denominator
         

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
