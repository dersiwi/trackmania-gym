"""Implementation of a custom feature extractor
https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#multiple-inputs-and-dictionary-observations"""
import gymnasium as gym
import torch 
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TMN_Extractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for the TrackMania environment observation space.
    This extractor is designed to work with observations containing both structured data 
    (e.g., SimStateData from TMInterface) and image inputs. It constructs a dedicated 
    feature extractor for each key in the observation space

    All extracted features are then concatenated and passed through an optional combined 
    MLP (not shown here, but can be added after this module).

    Note: This implementation is purpose-built for use with SimStateData and image inputs. 
    It does not handle edge cases and is not designed with general modularity in mind.

    :param observation_space: Gym Dict space describing the full observation structure.
    :param vision_model: A neural network used to extract features from image observations.
    :param vision_model_out_dim: The dimensionality of the feature vector produced by the vision model.
    :param normalized_image: If True, assumes that image inputs are already normalized.
"""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        vision_model,
        vision_model_out_dim : int = 128,
        normalized_image: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim=1)

        extractors: dict[str, nn.Module] = {}
        #  pytroch ModuleDict cant handel dots in key so we remove the dots but later we must know what the orginal key was 
        self.key_mappings: dict[str,str] = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                extractors[key] = vision_model
                total_concat_size += vision_model_out_dim
                self.key_mappings[key] = key
            else:
                """
                TODO do we really need to project down ? 
                Should our final feature vector be not that long ? -> could leed to not good training  
                see https://bmild.github.io/fourfeat/index.html
                """
                # pytroch ModuleDict cant handel dots in key 
                new_key = key.replace('.',"")
                self.key_mappings[new_key] = key
                if len(subspace.shape) == 1 and subspace.shape[0] > 50 :
                    extractors[new_key] = nn.Linear(subspace.shape[0], subspace.shape[0] //4)
                    total_concat_size += subspace.shape[0] //4
                elif len(subspace.shape) <= 1:
                    extractors[new_key] = nn.Identity()
                    total_concat_size += gym.spaces.utils.flatdim(subspace)
                else:
                    extractors[new_key] = nn.Flatten()
                    total_concat_size += gym.spaces.utils.flatdim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            tensor = extractor(observations[self.key_mappings[key]])
            if key == "image" :
                tensor = tensor[0]
            tensor = torch.tensor(tensor, dtype= torch.float)
            if len(tensor.shape) < 1 :
                tensor = tensor.unsqueeze(0)
            encoded_tensor_list.append(tensor)
        return torch.cat(encoded_tensor_list, dim=-1)
