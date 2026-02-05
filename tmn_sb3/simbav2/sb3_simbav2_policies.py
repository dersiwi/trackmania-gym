from typing import Any, Optional, Union

import numpy as np
import torch as th
from torch import nn
from gymnasium import spaces


from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.sac.policies import SACPolicy, Actor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)

from .simbav2_networks import SimbaV2Actor, SimbaV2Critic


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
        # TODO: look for the defautl values in the paper
        num_blocks: int = 4,
        hidden_features: int = 256,
        scaler_init: float = 1.0,
        scaler_scale: float = 1.0,
        alpha_init: float = 0.1,
        alpha_scale: float = 0.1,
        c_shift: float = 0.0,
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
        # TODO: check which default values the paper uses
        num_blocks: int = 4,
        hidden_features: int = 256,
        scaler_init: float = 1.0,
        scaler_scale: float = 1.0,
        alpha_init: float = 0.1,
        alpha_scale: float = 0.1,
        c_shift: float = 0.0,
        num_bins: int = 100,
        min_v: float = -200.0,
        max_v: float = 200.0,
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
        for idx in range(n_critics):
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
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )
