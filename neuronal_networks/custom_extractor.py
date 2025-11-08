from functools import partial
from typing import Callable, Union, List, Optional, Dict, Type, Tuple

import gymnasium as gym
from sympy.strategies.rl import subs
import torch 
from torch import nn
import numpy as np

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.preprocessing import preprocess_obs, is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor,MlpExtractor, create_mlp
from stable_baselines3.common.utils import get_device 

def build_vision_model(
    vision_model_cls: type[nn.Module],
    subspace: gym.Space,
    out_dim: int,
    device: str,
) -> nn.Module:
    """
    Helper function to instantiate and validate a vision model.

    Args:
        vision_model_cls: A PyTorch nn.Module *class* (not instance).
        subspace: The image observation space for this key.
        out_dim: Expected output dimension of the vision model.
        device: Device to move the model to.

    Returns:
        Instantiated vision model (nn.Module) placed on the given device.
    """
    # Check that the vision_model is indeed a subclass of nn.Module
    if not (isinstance(vision_model_cls, type) and issubclass(vision_model_cls, nn.Module)):
         raise TypeError("vision_model must be a PyTorch nn.Module class, not an instance.")
    # Instantiate the model
    model = vision_model_cls().to(device)

    # Check output shape
    dummy_input = torch.zeros(1, *subspace.shape, device=device)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    if dummy_output.ndim != 2:
        raise ValueError(
            f"Vision model output must have shape (N, D), but got {dummy_output.shape}"
        )

    vision_model_out_dim = dummy_output.shape[1]
    if vision_model_out_dim != out_dim:
        raise ValueError(
            f"Vision model output dimension ({vision_model_out_dim}) "
            f"does not match expected out_dim ({out_dim})."
        )

    return model


class TMN_Dict_Extractor(BaseFeaturesExtractor):
    """
    Implementation of a custom feature extractor 
    https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#multiple-inputs-and-dictionary-observations

    Combined feature extractor for the TrackMania environment observation space.
    This extractor is designed to work with observations containing  only vectors and image inputs. 
    It constructs a dedicated feature extractor for each key in the observation space

    All extracted features are then concatenated and passed through an optional combined 
    MLP (not shown here, but can be added after this module).

    :param observation_space: Gym Dict space describing the full observation structure.
    :param vision_model: A neural network used to extract features from image observations.
    :param out_dim: The number of dimension each extractor should project on to
    :param normalized_image: If True, assumes that image inputs are already normalized.        
    :param float_model (list[int]): Optional list defining MLP layer sizes for vector inputs.
    :param activation_fn (type[nn.Module]): Activation function class for MLPs.
    :param last_activation_fn (type[nn.Module]): Activation for final MLP layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        vision_model: type[torch.nn.Module],
        out_dim:int = 64,
        device= "cpu",
        normalized_image: bool = False,
        float_model: list[int] = None,
        activation_fn: type[nn.Module]= nn.ReLU,
        last_activation_fn: type[nn.Module]= nn.Tanh ) -> None:

        super().__init__(observation_space, features_dim=1)

        total_concat_size = 0
        extractors: dict[str, nn.Module] = {}
        for key, subspace in  observation_space.spaces.items(): 

            if is_image_space(subspace,check_channels= True, normalized_image = normalized_image):
                vision_model = build_vision_model(vision_model_cls= vision_model,subspace= subspace,out_dim= out_dim,device = device)  
                extractors[key] = vision_model
                total_concat_size += out_dim
            else:
                if float_model:
                    model_list = create_mlp(
                        input_dim= subspace.shape[0],
                        output_dim= out_dim,
                        net_arch= float_model
                        )
                else:

                    hidden_dim = subspace.shape[0] // 2 if subspace.shape[0] > out_dim else subspace.shape[0] * 2
                    model_list = [
                            nn.Linear(subspace.shape[0], hidden_dim, device=device),
                            nn.Linear(hidden_dim, out_dim, device=device)
                            ]
                extractors[key] = nn.Sequential(*model_list)
                total_concat_size += out_dim

        # Update the features dim manually
        self._features_dim = total_concat_size
        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            tensor = extractor(observations[key])
            encoded_tensor_list.append(tensor)
         # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)

class AsyncMLPExtractor(nn.Module):
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
    """ Example of Non-shared features extractor for actor critic style policies inspired by https://github.com/DLR-RM/stable-baselines3/issues/1066#issuecomment-1246866844"""
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
