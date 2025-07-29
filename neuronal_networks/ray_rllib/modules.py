import gymnasium as gym
import torch
from torch import nn

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.core.columns import Columns

from neuronal_networks.custom_extractor import TMN_Extractor
class TMNFActorCriticModule(TorchRLModule):
    """
    Implements a customizable Actor-Critic neural network module for use with Ray RLlib's
    Actor-Critic style algorithms (e.g., PPO, A2C).

    This module combines a feature extractor, a policy head (actor), and a value head (critic)
    into a single `TorchRLModule` to process observations, predict actions, and estimate
    state values.
    """

    @override(TorchRLModule)
    def setup(self):
         # Retrieve parameters from model_config
        self.vision_model_class = self.model_config.get("vision_model_class", None)
        self.share_feature_extractor = self.model_config.get("share_feature_extractor", True) 
        self.extractor_per_component_dim = self.model_config.get("extractor_per_component_dim", 64)
        self.float_model_layers = self.model_config.get("float_model_layers", [64, 32]) 
        self.activation_fn = self.model_config.get("activation_fn", nn.ReLU)
        self.last_activation_fn = self.model_config.get("last_activation_fn", nn.Tanh)
        self.normalized_image = self.model_config.get("normalized_image", False)
        self.vision_model_params = self.model_config.get("vision_model_params", {})
        self.device =  self.model_config.get("device", "cuda")

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

        # Instantiate the primary feature extractor (consistent naming: `feature_extractor`)
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

        if self.share_feature_extractor: 
            self.pi_features_extractor = self.feature_extractor 
            self.vf_features_extractor = self.feature_extractor 
        else:
            # If not sharing, both policy and value need their own independent extractors
            self.pi_features_extractor = self.feature_extractor
            
            separate_vision_model_instance = None
            if "image" in self.observation_space.spaces:
                 separate_vision_model_instance = self.vision_model_class(
                    out_dim=self.extractor_per_component_dim,
                    img_shape=self.observation_space.spaces["image"].shape,
                    **self.vision_model_params
                ).to(self.device)

            self.vf_features_extractor = TMN_Extractor( 
                observation_space=self.observation_space,
                vision_model=separate_vision_model_instance,
                out_dim=self.extractor_per_component_dim,
                device=self.device,
                normalized_image=self.normalized_image,
                float_model=self.float_model_layers,
                activation_fn=self.activation_fn,
                last_activation_fn=self.last_activation_fn
            )
        
        final_feature_dim_for_heads = self.feature_extractor._features_dim 

        # Policy head (actor)
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.policy_head = nn.Linear(final_feature_dim_for_heads, self.action_space.n).to(self.device)
        elif isinstance(self.action_space, gym.spaces.Box):
            self.policy_mean = nn.Linear(final_feature_dim_for_heads, self.action_space.shape[0]).to(self.device)
            self.policy_log_std = nn.Parameter(torch.zeros(self.action_space.shape[0])).to(self.device) 
        else:
            raise ValueError(f"Unsupported action space: {self.action_space}")

        # Value head (critic)
        self.value_head = nn.Linear(final_feature_dim_for_heads, 1).to(self.device)

        # Sanity check / verification: Check for parameters after setup 
        print(f"  Verifying parameters after TMNFActorCriticModule setup:")
        has_params = False
        for name, param in self.named_parameters():
            print(f"    - Found parameter: {name}, shape: {param.shape}, device: {param.device}")
            has_params = True
        
        if not has_params:
            print(f"    ERROR: No trainable parameters found in TMNFActorCriticModule after setup!")
            raise RuntimeError("TMNFActorCriticModule failed to register any trainable parameters. Optimizer cannot be created.")
        else:
            print(f"    SUCCESS: Trainable parameters found in TMNFActorCriticModule.")

    @override(TorchRLModule)
    def _forward_inference(self, batch, **kwargs):
        obs =  batch[Columns.OBS]
        policy_features = self.pi_features_extractor(obs) 
        value_features = self.vf_features_extractor(obs) 
        
        # Policy head
        if isinstance(self.action_space, gym.spaces.Discrete):
            action_logits = self.policy_head(policy_features)
        elif isinstance(self.action_space, gym.spaces.Box):
            action_mean = self.policy_mean(policy_features)
            action_log_std = self.policy_log_std.expand_as(action_mean)
            action_logits = (action_mean, action_log_std)
        
        # Value head
        values = self.value_head(value_features)
        
        return {
            Columns.ACTION_DIST_INPUTS: action_logits,
            Columns.VF_PREDS: values,
        }

    @override(TorchRLModule)
    def _forward_exploration(self, batch, **kwargs):
        obs =  batch[Columns.OBS]
        return self._forward_inference({Columns.OBS: obs}, **kwargs)

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs): 
        obs =  batch[Columns.OBS]

        policy_features = self.pi_features_extractor(obs)
        value_features = self.vf_features_extractor(obs) 

        if isinstance(self.action_space, gym.spaces.Discrete):
            action_logits = self.policy_head(policy_features)
        elif isinstance(self.action_space, gym.spaces.Box):
            action_mean = self.policy_mean(policy_features)
            action_log_std = self.policy_log_std.expand_as(action_mean)
            action_logits = (action_mean, action_log_std)
            
        values = self.value_head(value_features)

        return {
            Columns.ACTION_DIST_INPUTS: action_logits,
            Columns.VF_PREDS: values,
            "latent_policy_features": policy_features,
            "latent_value_features": value_features,
        }