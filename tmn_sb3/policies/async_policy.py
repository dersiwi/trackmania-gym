from functools import partial
from typing import Callable, Union, List, Optional, Dict, Type, Tuple, Any

import gymnasium as gym
import torch 
from torch import nn
import numpy as np

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.preprocessing import preprocess_obs 
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device 

class AsyncMLPExtractor(nn.Module):
    """
    This module defines separate MLP architectures for the actor and critic networks,
    designed for asynchronous setups where the two networks may run or update
    independently rather than sharing parameters or layers.

    Despite the name, this class is **not** a Stable-Baselines3 (SB3) feature extractor.  
    It only provides the MLP backbones for asynchronous actor–critic architectures.

    Args:
        feature_dim (Union[int, dict[str, int]]): 
            The dimensionality of the input features.  
            If an integer, the same input size is used for both actor and critic.  
            If a dictionary, it should contain separate keys "pi" (Actor) and "vf" (Critic)
            specifying their respective input dimensions.

        net_arch (Union[list[int], dict[str, list[int]]]): 
            The network architecture configuration.  
            If a list of integers, it defines a shared hidden layer structure.
            If a dictionary, it should contain separate keys "pi" (Actor) and "vf" (Critic)
            for their individual MLP layer sizes.

        activation_fn (type[nn.Module]): 
            The activation function to use (e.g., `nn.ReLU`, `nn.Tanh`, `nn.LeakyReLU`).

        device (Union[torch.device, str], optional): 
            The device on which to place the module (e.g., `"cpu"`, `"cuda"`, or `"auto"`).  
            Defaults to `"auto"`.
    """
    def __init__(
            self,
            feature_dim: Union[int, dict[str, int]],
            net_arch: Union[list[int], dict[str, list[int]]],
            activation_fn: type[nn.Module],
            device: Union[torch.device, str] = "auto",
            ):
        super().__init__()  # skip MlpExtractor's __init__, we override

        device = get_device(device)

        # Handle different input dimensions for actor (pi) and critic (vf)
        if isinstance(feature_dim, int):
            actor_input_dim = critic_input_dim = feature_dim
        else:
            actor_input_dim = feature_dim.get("pi", 0)
            critic_input_dim = feature_dim.get("vf", 0)

        if isinstance(net_arch, dict):
            pi_layers_dims = net_arch.get("pi", [])
            vf_layers_dims = net_arch.get("vf", [])
        else:
            pi_layers_dims = vf_layers_dims = net_arch

        # Build policy network
        policy_net = []
        last_layer_dim_pi = actor_input_dim
        for layer_size in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, layer_size))
            policy_net.append(activation_fn())
            last_layer_dim_pi = layer_size

        # Build value network
        value_net = []
        last_layer_dim_vf = critic_input_dim
        for layer_size in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, layer_size))
            value_net.append(activation_fn())
            last_layer_dim_vf = layer_size

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)

class AsyncActorCriticPolicy(ActorCriticPolicy):
    """
    Asynchronous Actor–Critic policy with non-shared feature extractors.

    This class implements a variant of the classic actor–critic architecture
    where the **actor** and **critic** use independent feature extractors
    (`policy_features_extractor` and `value_features_extractor`), enabling
    fully asynchronous processing and representation learning.

    The approach is inspired by:
        https://github.com/DLR-RM/stable-baselines3/issues/1066#issuecomment-1246866844

    Args:
        observation_space (gym.spaces.Space):
            The observation space of the environment.

        action_space (gym.spaces.Space):
            The action space of the environment.

        lr_schedule (Callable[[float], float]):
            Learning rate schedule function. The callable takes the remaining progress
            (from 1.0 to 0.0) and returns the learning rate to use.

        policy_features_extractor (BaseFeaturesExtractor):
            The feature extractor used by the actor (policy) network.

        value_features_extractor (BaseFeaturesExtractor):
            The feature extractor used by the critic (value) network.

        net_arch (Optional[List[Union[int, Dict[str, List[int]]]]], optional):
            Network architecture specification for the MLP extractor.
            Can be a shared list of layer sizes, or a dict with separate `"pi"` and `"vf"` keys.
            Defaults to ``None``.

        activation_fn (Type[nn.Module], optional):
            Activation function class to use in the MLP extractor. 
            Defaults to ``nn.Tanh``.

        *args:
            Additional positional arguments passed to ``ActorCriticPolicy``.

        **kwargs:
            Additional keyword arguments passed to ``ActorCriticPolicy``.
    """
    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Callable[[float], float],
            policy_features_extractor: BaseFeaturesExtractor,
            value_features_extractor: BaseFeaturesExtractor,
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            *args,
            **kwargs,
            ):
        super(AsyncActorCriticPolicy, self).__init__(
                observation_space,
                action_space,
                lr_schedule,
                net_arch,
                activation_fn,
                # Pass remaining arguments to base class
                *args,
                **kwargs,
                )

        # non-shared features extractors for the actor and the critic
        self.policy_features_extractor:BaseFeaturesExtractor = policy_features_extractor
        self.value_features_extractor:BaseFeaturesExtractor = value_features_extractor


        self.features_dim = {'pi': self.policy_features_extractor.features_dim,
                             'vf': self.value_features_extractor.features_dim}

        # NOTE: if the 2 features dims are different, your mlp_extractor must be able
        # to acceppt such dict AND ALSO an int, because the mlp_extractor will be first
        # created with wrong features_dim (coming from wrong, default, feratures extractor) which is an int.
        # Furthermore, note that with 2 different features dims the mlp_extractor cannot have shared layers.

        delattr(self, "features_extractor")  # remove the shared features extractor

        # Disable orthogonal initialization (if you want, otherwise comment it)
        self.ortho_init = False

        # The super-constructor calls a '_build' method that creates the network and the optimizer.
        # The problem is that it does so using a default features extractor, and not the ones just created,
        # therefore we need to re-create the mlp_extractor and the optimizer
        # (that otherwise would have used obsolete features_dims and parameters).
        self._rebuild(lr_schedule)

        self.policy_obs: list[str] = list(self.policy_features_extractor._observation_space.keys())
        self.value_obs: list[str] = list(self.value_features_extractor._observation_space.keys())

    def _rebuild(self, lr_schedule: Schedule) -> None:
        """ Re-creates the mlp_extractor and the optimizer for the model.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        # action_net and value_net as created in the '_build' method are OK,
        # no need to recreate them.

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                    self.policy_features_extractor: np.sqrt(2),
                    self.value_features_extractor: np.sqrt(2),
                    self.mlp_extractor: np.sqrt(2),
                    self.action_net: 0.01,
                    self.value_net: 1,
                    }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _build_mlp_extractor(self) -> None:

        self.mlp_extractor = AsyncMLPExtractor(   
                                               feature_dim= self.features_dim,
                                               net_arch=self.net_arch,
                                               activation_fn=self.activation_fn,
                                               device=self.device,)

    def extract_features(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :return: the output of the feature extractor(s)
        """
        assert self.policy_features_extractor is not None and self.value_features_extractor is not None
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        policy_obs = {k: v for k, v in preprocessed_obs.items() if k in self.policy_obs}
        value_obs = {k: v for k, v in preprocessed_obs.items() if k in self.value_obs}
        policy_features = self.policy_features_extractor(policy_obs)
        value_features = self.value_features_extractor(value_obs)
        return policy_features, value_features

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        policy_features, value_features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(policy_features)
        latent_vf = self.mlp_extractor.forward_critic(value_features)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        policy_features, value_features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(policy_features)
        latent_vf = self.mlp_extractor.forward_critic(value_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def get_distribution(self, obs: torch.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation
        :return: the action distribution.
        """
        policy_features, _ = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(policy_features)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        _, value_features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(value_features)
        return self.value_net(latent_vf)

### just some helper functions ###

from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from neuronal_networks.extractors import make_tmn_extractor

def build_async_actor_critic_policy(
    observation_space: spaces.Dict,
    actor_observations: List[str],
    critic_observations: List[str],
    actor_extractor_kwargs: Dict[str, Any],
    critic_extractor_kwargs: Dict[str, Any],
    net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
    activation_fn: Type[nn.Module] = nn.ReLU,
    normalize_images: bool = True,
) -> Tuple[Type[BasePolicy], Dict[str, Any]]:
    """
    Build an asynchronous actor–critic policy configuration for SB3.

    This helper isolates the creation of separate actor/critic feature extractors
    and returns everything needed to instantiate an AsyncActorCriticPolicy.

    Args:
        observation_space (spaces.Dict):
            Full observation space (must be a Dict space).
        actor_observations (list[str]):
            Keys of observations used by the actor network.
        critic_observations (list[str]):
            Keys of observations used by the critic network.
        actor_extractor_kwargs (dict):
            Keyword arguments passed to make_tmn_extractor() for the actor.
        critic_extractor_kwargs (dict):
            Keyword arguments passed to make_tmn_extractor() for the critic.
        net_arch (list[int] | dict[str, list[int]], optional):
            MLP architecture for actor/critic networks.
        activation_fn (type[nn.Module]):
            Activation function used inside the MLP extractor.
        normalize_images (bool):
            Whether to normalize image inputs.

    Returns:
        tuple[Type[AsyncActorCriticPolicy], dict]:
            A tuple containing the policy class and its kwargs dictionary.
    """

    if not isinstance(observation_space, spaces.Dict):
        raise ValueError("Async actor–critic requires a gym.spaces.Dict observation space.")

    if actor_observations is None or critic_observations is None:
        raise ValueError("You must specify both actor_observations and critic_observations.")

    actor_space = spaces.Dict({
        k: v for k, v in observation_space.spaces.items() if k in actor_observations
    })
    critic_space = spaces.Dict({
        k: v for k, v in observation_space.spaces.items() if k in critic_observations
    })

    policy_features_extractor = make_tmn_extractor(
        observation_space=actor_space,
        **actor_extractor_kwargs,
    )

    value_features_extractor = make_tmn_extractor(
        observation_space=critic_space,
        **critic_extractor_kwargs,
    )

    return AsyncActorCriticPolicy, dict(
        policy_features_extractor=policy_features_extractor,
        value_features_extractor=value_features_extractor,
        net_arch=net_arch,
        activation_fn=activation_fn,
        normalize_images=normalize_images,
    )
