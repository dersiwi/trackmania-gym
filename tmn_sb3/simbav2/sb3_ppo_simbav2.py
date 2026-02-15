from typing import Optional, Union, Any

import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import (
    CategoricalDistribution,
    DiagGaussianDistribution,
    SquashedDiagGaussianDistribution,
    Distribution,
)
from stable_baselines3.common.preprocessing import get_action_dim

from .simbav2_networks import SimbaV2Actor, SimbaV2Critic


class PPOSimbaV2Policy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = BaseFeaturesExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        actor_kwargs: dict[str, Any] | None = None,
        critic_kwargs: dict[str, Any] | None = None,
        min_v: float = -100.0,
        max_v: float = 100.0,
        num_bins: int = 255,
    ):
        # skip the parents init since we will do things vaguely different
        super(ActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )

        # copied from the parent class
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.share_features_extractor = share_features_extractor
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        if self.share_features_extractor:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.features_extractor
        else:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.make_features_extractor()

        self.log_std_init = log_std_init
        dist_kwargs = None

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        action_dim = get_action_dim(self.action_space)
        # Action distribution. TODO: figure out if its ok to use the squashed version or do we need the normal diag gauss
        self.action_dist = SquashedDiagGaussianDistribution(action_dim)

        assert self.features_dim is not None, "At this stage the feature extractor dim should be set"

        assert actor_kwargs is not None, "actor_kwargs must be provided"
        assert critic_kwargs is not None, "critic_kwargs must be provided"
        self.actor_kwargs = actor_kwargs
        self.critic_kwargs = critic_kwargs

        self.actor_kwargs["in_features"] = self.features_dim
        self.critic_kwargs["in_features"] = self.features_dim

        # HL-Gauss Support Setup
        self.min_v = min_v
        self.max_v = max_v
        self.num_bins = num_bins
        self.register_buffer("support", th.linspace(self.min_v, self.max_v, self.num_bins))

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = SimbaV2Actor(**self.actor_kwargs)
        self.critic = SimbaV2Critic(**self.critic_kwargs)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]


    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        if isinstance(self.action_dist, DiagGaussianDistribution):
            mean, log_std = self.actor(latent_pi)
            return self.action_dist.proba_distribution(mean,log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            mean = self.actor(latent_pi)
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean)
        else:
            raise ValueError("Invalid action distribution")

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value, log probability of the action, log probs of the values
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # Evaluate the values for the given observations
        values_log_probs = self.critic(latent_vf)
        value_probs = th.exp(values_log_probs)
        # values are just the expected value (dot product of probs and support)
        # Shape goes from (batch_size, num_bins) -> (batch_size,)
        values = th.sum(value_probs * self.support, dim=-1)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        actions_log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, actions_log_prob, values_log_probs
