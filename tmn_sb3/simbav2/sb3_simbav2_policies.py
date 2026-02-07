from typing import Any, Optional, Union

import numpy as np
import torch as th
from torch import nn
from gymnasium import spaces

from stable_baselines3.sac import SAC
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.sac.policies import SACPolicy, Actor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)

from .simbav2_networks import SimbaV2Actor, SimbaV2Critic
from .normalizers import SimbaVecNormalize


class SACSimbaV2Actor(Actor):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: nn.Module,
        features_dim: int,
        # only keep this network stuff for now since we dont know if sb3 really needs them
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2,
        normalize_images: bool = True,
        ### SimbaV2 specific args ###
        num_blocks: int = 1,
        hidden_features: int = 128,
        scaler_init: float = np.sqrt(2/128), 
        scaler_scale: float = np.sqrt(2/128),
        alpha_init: float = 1/2,
        alpha_scale: float = np.sqrt(1/128),
        c_shift: float = 3.0,
        gain: float = 1.0,
        **kwargs,
    ):
        # call only the BasePolicy constructor
        super(Actor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.use_sde = use_sde
        self.log_std_init = log_std_init
        self.full_std = full_std
        self.use_expln = use_expln
        self.clip_mean = clip_mean
        # Simba specific attributes
        self.num_blocks = num_blocks
        self.hidden_features = hidden_features
        self.scaler_init = scaler_init
        self.scaler_scale = scaler_scale
        self.alpha_init = alpha_init
        self.alpha_scale = alpha_scale
        self.c_shift = c_shift
        self.gain = gain

        action_dim = get_action_dim(self.action_space)
        self.simbav2_core = SimbaV2Actor(
            num_blocks=num_blocks,
            in_features=features_dim,
            hidden_features=hidden_features,
            action_dim=action_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            alpha_init=alpha_init,
            alpha_scale=alpha_scale,
            c_shift=c_shift,
            gain=gain,
        )

        self.action_dist = SquashedDiagGaussianDistribution(action_dim)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        """Expose the parameters to the SB3 saver."""
        # Start with the BasePolicy parameters (obs_space, action_space, etc.)
        data = super(Actor, self)._get_constructor_parameters()

        # Update with everything needed to re-run __init__
        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                clip_mean=self.clip_mean,
                # Simba Specific
                num_blocks=self.num_blocks,
                hidden_features=self.hidden_features,
                scaler_init=self.scaler_init,
                scaler_scale=self.scaler_scale,
                alpha_init=self.alpha_init,
                alpha_scale=self.alpha_scale,
                c_shift=self.c_shift,
                gain=self.gain,
            )
        )
        return data

    def get_action_dist_params(self, obs: PyTorchObs) -> tuple[th.Tensor, th.Tensor, dict[str, th.Tensor]]:
        features = self.extract_features(obs=obs, features_extractor=self.features_extractor)
        mean, log_std = self.simbav2_core(features)
        return mean, log_std, {}


class SACSimbaV2Critic(ContinuousCritic):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        ### Simba specific args ###
        num_blocks: int = 2,
        hidden_features: int = 512,
        scaler_init: float = np.sqrt(2/512),
        scaler_scale: float = np.sqrt(2/512),
        alpha_init: float = 1/3,
        alpha_scale: float = np.sqrt(2/512),
        c_shift: float = 3.0,
        num_bins: int = 101,
        min_v: float = -5,
        max_v: float = 5,
        gain: float = 1.0,
        **kwargs,
    ):
        # call only the super method of the parent class
        super(ContinuousCritic, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)
        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.num_blocks = num_blocks
        self.hidden_features = hidden_features
        self.scaler_init = scaler_init
        self.scaler_scale = scaler_scale
        self.alpha_init = alpha_init
        self.alpha_scale = alpha_scale
        self.c_shift = c_shift
        self.num_bins = num_bins
        self.min_v = min_v
        self.max_v = max_v
        self.gain = gain

        self.q_networks: list[nn.Module] = []
        for idx in range(n_critics):  # NOTE: By default, clipping is deactivated (set to infinity) to match SimBa,
            # which does not apply observation clipping. Clipping is only enabled if
            # non-default values are provided (i.e., something other than np.inf).
            q_net = SimbaV2Critic(
                num_blocks=num_blocks,
                in_features=features_dim + action_dim,
                hidden_features=hidden_features,
                scaler_init=scaler_init,
                scaler_scale=scaler_scale,
                alpha_init=alpha_init,
                alpha_scale=alpha_scale,
                c_shift=c_shift,
                num_bins=num_bins,
                min_v=min_v,
                max_v=max_v,
                gain=gain,
            )
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)


class SACSimbaV2Policy(SACPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        actor_kwargs: dict[str, Any] | None = None,
        critic_kwargs: dict[str, Any] | None = None,
    ):
        super(SACPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [256, 256]

        if isinstance(net_arch, list):
            actor_arch, critic_arch = net_arch, net_arch
        else:
            actor_arch = net_arch.get("pi", [])
            critic_arch = net_arch.get("qf", [])

        self.actor_kwargs = actor_kwargs if actor_kwargs is not None else {}
        self.critic_kwargs = critic_kwargs if critic_kwargs is not None else {}

        # NOTE: in_features will be set when calling the make_actor/make_critic methods since it need to retrieve that info
        # from the features_extractor
        self.actor_kwargs.update(
            {
                "observation_space": self.observation_space,
                "action_space": self.action_space,
                "net_arch": actor_arch,
                "activation_fn": activation_fn,
                "use_sde": use_sde,
                "log_std_init": log_std_init,
                "use_expln": use_expln,
                "clip_mean": clip_mean,
            }
        )

        self.critic_kwargs.update(
            {
                "observation_space": self.observation_space,
                "action_space": self.action_space,
                "net_arch": critic_arch,
                "activation_fn": activation_fn,
                "n_critics": n_critics,
                "share_features_extractor": share_features_extractor,
            }
        )

        self._build(lr_schedule)

    def make_actor(self, features_extractor: BaseFeaturesExtractor | None = None) -> SACSimbaV2Actor:
        kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return SACSimbaV2Actor(**kwargs).to(self.device)

    def make_critic(self, features_extractor: BaseFeaturesExtractor | None = None) -> SACSimbaV2Critic:
        kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return SACSimbaV2Critic(**kwargs).to(self.device)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        # We only add the parameters that __init__ actually expects
        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.actor_kwargs["use_sde"],
                log_std_init=self.actor_kwargs["log_std_init"],
                use_expln=self.actor_kwargs["use_expln"],
                clip_mean=self.actor_kwargs["clip_mean"],
                n_critics=self.critic_kwargs["n_critics"],
                share_features_extractor=self.critic_kwargs["share_features_extractor"],
                lr_schedule=self._dummy_schedule,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                # These dicts contain all the SimbaV2-specific blocks, dims, and scalers for the actor and the critic
                actor_kwargs=self.actor_kwargs,
                critic_kwargs=self.critic_kwargs,
            )
        )
        return data


class SACSimbaV2(SAC):
    def __init__(
        self,
        policy: Union[str, type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 0.0003,
        buffer_size: int = 1000000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        # NOTE: this is only a sanity check for us now so that we do not forget it
        assert isinstance(env, SimbaVecNormalize), (
            "SimBa explicitly relies on SimbaVecNormalize for RSNorm-style observation normalization."
        )
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            n_steps,
            ent_coef,
            target_update_interval,
            target_entropy,
            use_sde,
            sde_sample_freq,
            use_sde_at_warmup,
            stats_window_size,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )
