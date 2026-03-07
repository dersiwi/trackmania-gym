from typing import Optional

import numpy as np

from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn


class SimbaVecNormalize(VecNormalize):
    """
        SimbaV1: https://arxiv.org/pdf/2410.09754
        SimbaV2: https://arxiv.org/pdf/2502.15280

        VecNormalize wrapper implementing the observation (and optional reward)
        normalization used in SimBa (v1 & v2).

        Key design choice from SimBav1 on which also Simbav2 builds upon
        (from Simbav1 paper Section 7.1):
        - Observations and rewards should be stored *unnormalized* in the replay buffer.
        - Normalization is applied only when sampling batches for gradient updates.

    Stable-Baselines3 already follows this behavior internally: replay buffers
        store raw data, and normalization is applied during sampling via the
        VecNormalize wrapper. This class exists to explicitly align SB3 usage with
        SimBa's RSNorm setup.

        #NOTE::
        - SimBaV1 and SimBaV2 use the same observation normalization.

        - By default, clipping is deactivated (set to infinity) to match SimBa,
            which does not apply observation clipping. Clipping is only enabled if
            non-default values are provided (i.e., something other than np.inf).

        - SimBa does NOT normalize rewards; set `norm_reward=False` when following
          the paper exactly.
    """

    def __init__(
        self,
        venv: VecEnv,
        training: bool = True,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = np.inf,
        clip_reward: float = np.inf,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        g_max: Optional[float] = 10.0,
        norm_obs_keys: Optional[list[str]] = None,
    ):
        super().__init__(venv, training, norm_obs, norm_reward, clip_obs, clip_reward, gamma, epsilon, norm_obs_keys)

        self.g_r_max = 0.0  # Running max of absolute returns
        self.g_max = g_max

    # we dont normalize the obs/rewards and add them to the replay buffer instead we only
    # update the mean and var when step_wait is called and return unnormalized obs and rewards
    def step_wait(self) -> VecEnvStepReturn:
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, dones)

        where ``dones`` is a boolean vector indicating whether each element is new.
        """
        obs, rewards, dones, infos = self.venv.step_wait()
        assert isinstance(obs, (np.ndarray, dict))  # for mypy
        self.old_obs = obs
        self.old_reward = rewards

        if self.training and self.norm_obs:
            if isinstance(obs, dict) and isinstance(self.obs_rms, dict):
                for key in self.obs_rms.keys():
                    self.obs_rms[key].update(obs[key])
            else:
                self.obs_rms.update(obs)

        if self.training:
            self._update_reward(rewards)

        # Normalize the terminal observations
        for idx, done in enumerate(dones):
            if not done:
                continue
            if "terminal_observation" in infos[idx]:
                infos[idx]["terminal_observation"] = self.normalize_obs(infos[idx]["terminal_observation"])

        self.returns[dones] = 0
        return obs, rewards, dones, infos

    def _update_reward(self, reward: np.ndarray) -> None:
        super()._update_reward(reward)  # parent class already does G_t = gamma * G_{t-1} + r_t
        self.g_r_max = max(self.g_r_max, np.max(np.abs(self.returns)))  # Eq 18 from simbav2

    def normalize_reward(self, reward: np.ndarray) -> np.ndarray:
        if self.norm_reward:
            # Eq 19 from simbav2
            var_denominator = np.sqrt(self.ret_rms.var + self.epsilon)
            min_required_denominator = self.g_r_max / self.g_max
            denominator = max(var_denominator, min_required_denominator)
            reward = np.clip(reward / denominator, -self.clip_reward, self.clip_reward)
        # Note: we cast to float32 as it correspond to Python default float type
        # This cast is needed because `RunningMeanStd` keeps stats in float64
        return reward.astype(np.float32)


class OnPolicySimbaVecNormalize(VecNormalize):
    """
    Is just the normal VecNormalize but with the reward scaling/clipping from simba2
    """

    def __init__(
        self,
        venv: VecEnv,
        training: bool = True,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = np.inf,
        clip_reward: float = np.inf,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        g_max: Optional[float] = 10.0,
        norm_obs_keys: Optional[list[str]] = None,
    ):
        super().__init__(venv, training, norm_obs, norm_reward, clip_obs, clip_reward, gamma, epsilon, norm_obs_keys)

        self.g_r_max = 0.0  # Running max of absolute returns
        self.g_max = g_max


    def _update_reward(self, reward: np.ndarray) -> None:
        super()._update_reward(reward)  # parent class already does G_t = gamma * G_{t-1} + r_t
        self.g_r_max = max(self.g_r_max, np.max(np.abs(self.returns)))  # Eq 18 from simbav2

    def normalize_reward(self, reward: np.ndarray) -> np.ndarray:
        if self.norm_reward:
            # Eq 19 from simbav2
            var_denominator = np.sqrt(self.ret_rms.var + self.epsilon)
            min_required_denominator = self.g_r_max / self.g_max
            denominator = max(var_denominator, min_required_denominator)
            reward = np.clip(reward / denominator, -self.clip_reward, self.clip_reward)
        # Note: we cast to float32 as it correspond to Python default float type
        # This cast is needed because `RunningMeanStd` keeps stats in float64
        return reward.astype(np.float32)
