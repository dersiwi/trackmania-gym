from typing import Any, Dict, Optional
import gymnasium as gym
import torch
from torch import nn
from torchrl.modules import NoisyLinear

from ray.rllib.core.rl_module.apis import (
    TargetNetworkAPI,
    QNetAPI,
    TARGET_NETWORK_ACTION_DIST_INPUTS,
)

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.core.columns import Columns

from ray.rllib.core.learner.utils import make_target_network
from ray.rllib.utils.typing import TensorType

from neuronal_networks.custom_extractor import TMN_Extractor    

class TMNFDistDQNModule(TorchRLModule,QNetAPI,TargetNetworkAPI):

    def _build_mlp_with_noisy_heads(self, in_dim: int, hidden_layers: list, out_dim: int, activation_fn: nn.Module) -> nn.Sequential:
        """
        A helper method to build an MLP with NoisyLinear layers and a final output layer.
        """
        layers = []
        current_in_dim = in_dim
        for out_dim_mlp in hidden_layers:
            layers.append(NoisyLinear(current_in_dim, out_dim_mlp))
            layers.append(activation_fn())
            current_in_dim = out_dim_mlp
        
        # Add the final output layer, which is also a NoisyLinear layer
        layers.append(NoisyLinear(current_in_dim, out_dim))
        
        return nn.Sequential(*layers)
      
    @override(TorchRLModule)
    def setup(self):

         # Retrieve parameters from model_config
        self.vision_model_class = self.model_config.get("vision_model_class", None)
        self.extractor_per_component_dim = self.model_config.get("extractor_per_component_dim", 64)
        self.float_model_layers = self.model_config.get("float_model_layers", [64, 32]) 
        self.activation_fn = self.model_config.get("activation_fn", nn.ReLU)
        self.last_activation_fn = self.model_config.get("last_activation_fn", nn.Tanh)
        self.normalized_image = self.model_config.get("normalized_image", False)
        self.vision_model_params = self.model_config.get("vision_model_params", {})
        self.device =  self.model_config.get("device", "cuda")

        ### Building the feature extractor ###
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

        final_feature_dim = self.feature_extractor._features_dim 

        ### Q-Network related things ###
        self.num_atoms = self.model_config.get("num_atoms", 51)
        self.v_min = self.model_config.get("v_min", -10.0)
        self.v_max = self.model_config.get("v_max", 10.0)
        
        # dueling dqn 
        value_head_layers = self.model_config.get("value_head_layers", [256])
        advantage_head_layers = self.model_config.get("advantage_head_layers", [256])

        self.value_net = self._build_mlp_with_noisy_heads(
            in_dim=final_feature_dim,
            hidden_layers=value_head_layers,
            out_dim=self.num_atoms,
            activation_fn=self.activation_fn
        ).to(self.device)

        self.advantage_net = self._build_mlp_with_noisy_heads(
            in_dim=final_feature_dim,
            hidden_layers=advantage_head_layers,
            out_dim=self.action_space.n * self.num_atoms,
            activation_fn=self.activation_fn
        ).to(self.device)

        # Sanity check / verification: Check for parameters after setup 
        self.sanity_check()

    @override(QNetAPI)
    def compute_q_values(self, batch: Dict[str, TensorType]) -> Dict[str, TensorType]:
        """
        Computes Q-value distributions using a Dueling, Distributional architecture.

        This implementation combines the outputs of the value and advantage heads
        to form a final Q-value distribution, as required by the QNetAPI.
        """
        obs = batch[Columns.OBS]
        features = self.feature_extractor(obs)

        # Compute value stream distribution logits.
        value_logits = self.value_net(features)
        # Reshape to (batch_size, 1, num_atoms) for broadcasting.
        value_logits = value_logits.unsqueeze(dim=1) 

        # Compute advantage stream distribution logits.
        advantage_logits = self.advantage_net(features)
        # Reshape to (batch_size, num_actions, num_atoms).
        advantage_logits = advantage_logits.view(-1, self.action_space.n, self.num_atoms)
        
        # Dueling Aggregation: Combine value and advantage distributions.
        mean_advantage_logits = torch.mean(advantage_logits, dim=1, keepdim=True)
        q_logits_per_atom = value_logits + advantage_logits - mean_advantage_logits

        # Compute probabilities from logits.
        q_probs_per_atom = nn.functional.softmax(q_logits_per_atom, dim=-1)
        
        # Compute the mean Q-value as the sum of (probability * atom_value).
        atoms = torch.linspace(self.v_min, self.v_max, self.num_atoms, device=self.device)
        q_values = torch.sum(q_probs_per_atom * atoms, dim=-1)
        
        return {
            "qf_preds": q_values,
            "qf_logits": q_logits_per_atom,
            "qf_probs": q_probs_per_atom,
            "atoms": atoms,
        }

    @override(QNetAPI)
    def compute_advantage_distribution(
        self,
        batch: Dict[str, TensorType],
    ) -> Dict[str, TensorType]:
        """
        Computes the advantage distribution.

        For a Dueling Distributional Q-network, this implementation directly
        returns the Q-distribution by default, which is sufficient for the
        DQN learner.
        """
        return self.compute_q_values(batch)
    
    def sanity_check(self):
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