"""This is the equivalent to the feature extractor in sb3"""
from typing import Any, Dict, List
import torch.nn as nn

from gymnasium.spaces import Dict

from neuronal_networks.custom_extractor import TMN_Extractor
from ray.rllib.core.models.base import Encoder,ActorCriticEncoder 
from ray.rllib.core.models.configs import ModelConfig


class TMN_Encoder(Encoder):
    framework = "torch"
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Unpack config to get all the parameters
        self.observation_space = config.observation_space
        self.vision_model_class = config.vision_model_class
        self.extractor_per_component_dim = config.extractor_per_component_dim
        self.float_model_layers = config.float_model_layers
        self.activation_fn = config.activation_fn
        self.last_activation_fn = config.last_activation_fn
        self.normalized_image = config.normalized_image
        self.vision_model_params = config.vision_model_params
        self.device = config.device
        
        vision_model_instance = None
        if "image" in self.observation_space.spaces:
            if self.vision_model_class is None:
                raise ValueError("`vision_model_class` must be provided in `model_config` "
                                 "if 'image' is in the observation space.")
            vision_model_instance = self.vision_model_class(
                out_dim=self.extractor_per_component_dim,
                img_shape=self.observation_space.spaces["image"].shape,
                **self.vision_model_params
            ).to(self.device) 

        self.feature_extractor = TMN_Extractor(
            observation_space=self.observation_space,
            vision_model=vision_model_instance, 
            out_dim=self.extractor_per_component_dim,
            device=self.device, 
            normalized_image=self.normalized_image,
            float_model=self.float_model_layers,
            activation_fn=self.activation_fn,
            last_activation_fn=self.last_activation_fn
        )

    def _forward(self, input_dict: Dict[str, Any], **kwargs):
        # This is the core forward pass for a single stream.
        # It takes the input and passes it through the extractor.
        features = self.feature_extractor(input_dict)
        return {
            "encoder_out": features
        }
    

class TMNF_ActorCriticEncoder(ActorCriticEncoder):
    """An encoder that potentially holds two stateless encoders.

    This is a special case of Encoder that can either enclose a single,
    shared encoder or two separate encoders: One for the actor and one for the
    critic. The two encoders are of the same type, and we can therefore make the
    assumption that they have the same input and output specs.
    """

    framework = None

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        # Retrieve parameters from model_config
        self.observation_space =  config.observation_space
        self.vision_model_class = self.config.get("vision_model_class", None)
        self.extractor_per_component_dim = self.model_config.get("extractor_per_component_dim", 64)
        self.float_model_layers = self.model_config.get("float_model_layers", [64, 32]) 
        self.activation_fn = self.model_config.get("activation_fn", nn.ReLU)
        self.last_activation_fn = self.model_config.get("last_activation_fn", nn.Tanh)
        self.normalized_image = self.model_config.get("normalized_image", False)
        self.vision_model_params = self.model_config.get("vision_model_params", {})
        self.device =  self.model_config.get("device", "cuda")

        self.actor_observations = self.model_config.get("actor_obs", ["image","floats"])
        self.critic_observations = self.model_config.get("critic_obs", ["image","floats"])

        ### Building the feature extractor ###
        if config.shared:
            self.encoder = self._create_extractor(
                observations= self.observation_space,
                extractor_per_component_dim= self.extractor_per_component_dim,
                float_model_layers= self.float_model_layers,
                activation_fn= self.activation_fn,
                last_activation_fn = self.last_activation_fn,
                normalized_image= self.normalized_image,
                device= self.device
            )
        else:
            actor_obs_space = self._filter_observation_space(self.actor_observations)
            self.actor_encoder = self._create_extractor(
                observations= actor_obs_space,
                extractor_per_component_dim= self.extractor_per_component_dim,
                float_model_layers= self.float_model_layers,
                activation_fn= self.activation_fn,
                last_activation_fn = self.last_activation_fn,
                normalized_image= self.normalized_image,
                device= self.device
            )

            self.critic_encoder = None
            if not config.inference_only:
                critic_obs_space = self._filter_observation_space(self.actor_observations)
                self.critic_encoder = self._create_extractor(
                observations= critic_obs_space,
                extractor_per_component_dim= self.extractor_per_component_dim,
                float_model_layers= self.float_model_layers,
                activation_fn= self.activation_fn,
                last_activation_fn = self.last_activation_fn,
                normalized_image= self.normalized_image,
                device= self.device
            )

    def _create_extractor(self, observation_space: Dict ,extractor_per_component_dim:int, float_model_layers: List[int], activation_fn:nn.Module, last_activation_fn:nn.Module, normalized_image:bool, device:str):
        """Helper method to create a TMN_Extractor instance with all parameters passed directly."""

        vision_model_instance = None

        if "image" in  observation_space.spaces:

            if self.vision_model_class is None:
                raise ValueError("`vision_model_class` must be provided in `model_config` "
                                 "if 'image' is in the observation space.")
            
            vision_model_instance = self.vision_model_class(
                out_dim=self.extractor_per_component_dim,
                img_shape=self.observation_space.spaces["image"].shape,
                **self.vision_model_params
            ).to(self.device) 

        return TMN_Extractor(
            observation_space=self.observation_space,
            vision_model=vision_model_instance,
            out_dim=extractor_per_component_dim,
            device=device,
            normalized_image=normalized_image,
            float_model=float_model_layers,
            activation_fn=activation_fn,
            last_activation_fn=last_activation_fn
        )
    
    def _filter_observation_space(self, obs_keys: List[str]) -> Dict:
        """
        Helper method to create a filtered Dict observation space.
        """
        return Dict({k: self.observation_space[k] for k in obs_keys})
