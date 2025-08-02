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
        
        # dueling dqn 
        self.value = NoisyLinear(final_feature_dim,1)
        self.advantages = NoisyLinear(final_feature_dim,self.action_space.n)

        # distributional q-learning 
        self.num_atoms = self.model_config.get("num_atoms", 51)
        self.v_min = self.model_config.get("v_min", -10.0)
        self.v_max = self.model_config.get("v_max", 10.0)
        
        q_model_layers = self.model_config.get("q_model_layers", [256, 256]) 
        layers = []
        in_dim = final_feature_dim
        for out_dim in q_model_layers:
            layers.append(NoisyLinear(in_dim, out_dim))
            layers.append(self.activation_fn())
            in_dim = out_dim
        # Add the final output layer
        layers.append(NoisyLinear(in_dim, self.action_space.n * self.num_atoms))
        
        self.qnet = nn.Sequential(*layers).to(self.device)

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

    # ... (Your existing setup method)
    # You would need to add num_atoms, v_min, and v_max to your setup here.
    # For example:
    # self.num_atoms = self.model_config.get("num_atoms", 51)
    # self.v_min = self.model_config.get("v_min", -10.0)
    # self.v_max = self.model_config.get("v_max", 10.0)

    # In your Q-network setup, the final layer must output num_atoms * num_actions
    # self.qnet = NoisyLinear(in_dim, self.action_space.n * self.num_atoms)
    # and then reshape it later.

    @override(QNetAPI)
    def compute_q_values(self, batch: Dict[str, TensorType]) -> Dict[str, TensorType]:
        """
        Computes Q-value distributions (logits, probabilities) and mean Q-values.
        """
        obs = batch[Columns.OBS]
        
        # Pass the observation through the feature extractor
        features = self.feature_extractor(obs)
        
        # Pass features through the Q-network
        q_logits_per_atom = self.qnet(features)
        
        # Reshape the logits to (batch_size, num_actions, num_atoms)
        q_logits_per_atom = q_logits_per_atom.view(
            -1, self.action_space.n, self.num_atoms
        )
        
        # Compute probabilities from logits (softmax over the atoms dimension)
        q_probs_per_atom = nn.functional.softmax(q_logits_per_atom, dim=-1)
        
        # Compute the Q-value predictions (mean of the distribution)
        # This requires creating the tensor of atoms (the support of the distribution)
        atoms = torch.linspace(self.v_min, self.v_max, self.num_atoms, device=self.device)
        
        # Compute the mean Q-value as the sum of (probability * atom_value)
        q_values = torch.sum(q_probs_per_atom * atoms, dim=-1)
        
        return {
            "qf_preds": q_values,
            "qf_logits": q_logits_per_atom,
            "qf_probs": q_probs_per_atom,
            "atoms": atoms,
        }