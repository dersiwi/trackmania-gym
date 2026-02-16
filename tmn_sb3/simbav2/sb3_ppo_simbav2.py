import collections
from typing import Optional, Union, Any

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from stable_baselines3.ppo import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import (
    Distribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
)
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import (
    PyTorchObs,
    Schedule,
    GymEnv,
)
from stable_baselines3.common.utils import explained_variance

from .simbav2_networks import (
    SimbaV2Actor,
    SimbaV2DiscreteActor,
)
from .hl_gauss import HLGaussLoss, HLGaussCritic


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
        # Action distribution.
        self.action_dist = (
            DiagGaussianDistribution(action_dim)
            if isinstance(self.action_space, spaces.Box)
            else CategoricalDistribution(action_dim)
        )

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
        self.register_buffer("support", th.linspace(self.min_v, self.max_v, self.num_bins + 1))

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = (
            SimbaV2Actor(**self.actor_kwargs)
            if isinstance(self.action_space, spaces.Box)
            else SimbaV2DiscreteActor(**self.actor_kwargs)
        )

        # TODO: really check that the crtitic must output (batch_size,num_bins) and not (batch_size,num_bins+1)
        self.critic = HLGaussCritic(**self.critic_kwargs)

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
            return self.action_dist.proba_distribution(mean, log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            mean = self.actor(latent_pi)
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean)
        else:
            raise ValueError("Invalid action distribution")

    def transform_from_probs(self, probs: th.Tensor) -> th.Tensor:
        centers = (self.support[:-1] + self.support[1:]) / 2
        return th.sum(probs * centers, dim=-1)

    def get_scalar_values_and_logits(self, vf_features: th.Tensor):
        values_logits = self.critic(vf_features)
        value_probs = F.softmax(values_logits, dim=-1)
        assert th.allclose(value_probs.sum(dim=-1), th.ones(value_probs.size(0)), atol=1e-6)
        # values are just the expected value (dot product of probs and support)
        values = self.transform_from_probs(value_probs)  # Shape goes from (batch_size, num_bins) -> (batch_size,)
        return values, values_logits

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value, log probability of the action, log probs of the values
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        pi_features, vf_features = features

        # Evaluate the values for the given observations
        values, _ = self.get_scalar_values_and_logits(vf_features)
        distribution = self._get_action_dist_from_latent(pi_features)
        actions = distribution.get_actions(deterministic=deterministic)
        actions_log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, actions_log_prob

    def evaluate_actions(
        self, obs: PyTorchObs, actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        pi_features, vf_features = features
        distribution = self._get_action_dist_from_latent(pi_features)
        log_prob = distribution.log_prob(actions)
        values, values_logits = self.get_scalar_values_and_logits(vf_features)
        entropy = distribution.entropy()
        return values, log_prob, entropy, values_logits

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        pi_features, _ = features
        return self._get_action_dist_from_latent(pi_features)

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = super().extract_features(obs, self.vf_features_extractor)
        _, vf_features = features
        values, _ = self.get_scalar_values_and_logits(vf_features)
        return values

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)  # type: ignore[arg-type, return-value]

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                # --- NEW ARGUMENTS ADDED HERE ---
                actor_kwargs=self.actor_kwargs,
                critic_kwargs=self.critic_kwargs,
                min_v=self.min_v,
                max_v=self.max_v,
                num_bins=self.num_bins,
            )
        )
        return data


class SimbaV2PPO(PPO):
    def __init__(
        self,
        policy: Union[str, type[PPOSimbaV2Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 0.0003,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        num_bins: int = 101,
        min_v: float = -5.0,
        max_v: float = 5.0,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            n_steps,
            batch_size,
            n_epochs,
            gamma,
            gae_lambda,
            clip_range,
            clip_range_vf,
            normalize_advantage,
            ent_coef,
            vf_coef,
            max_grad_norm,
            use_sde,
            sde_sample_freq,
            rollout_buffer_class,
            rollout_buffer_kwargs,
            target_kl,
            stats_window_size,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )
        self.num_bins = num_bins
        self.max_v = max_v
        self.min_v = min_v
        self.bin_values = th.linspace(start=self.min_v, end=self.max_v, steps=self.num_bins, device=device)
        self.hl_gauss_loss = HLGaussLoss(
            min_value=self.min_v, max_value=self.max_v, num_bins=self.num_bins, sigma=1.0, device=device
        )

    # Just copy pasted the og sb3 ppo code, did not know how to do it cleaner
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy, values_logits = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                
                # Values loss 
                # the returns should be already clipped by the env wrapper 
                value_loss = self.hl_gauss_loss(values_logits,rollout_data.returns)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
