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

        total_concat_size = 0
        extractors: dict[str, nn.Module] = {}

        """
        pytorch ModuleDict cant handel dots in key so we remove 
        the dots but later we must know what the orginal key was 
        """  
        self.key_mappings: dict[str,str] = {}
        """
        Later the inputs will come in batches (b : batch size) and will be turned into flat vectors:
        image: (b,c,w,h) -> (b, vision_model_out_dim)
        scalars: (b) -> (b,1)
        lists: (b,length of list) -> (b,length of list) [depends on length of list]
        rotation_matrix: (b,3,3)  -> (b,9)
        """
        for key, subspace in observation_space.spaces.items():

            if key == "image":
                extractors[key] = vision_model
                total_concat_size += vision_model_out_dim
                self.key_mappings[key] = key
            else:
                # remove the dots for the pytorch ModuleDict but remember the actual key
                new_key = key.replace('.',"")
                self.key_mappings[new_key] = key

                # if subspace is a vector whith length > thresh (here 50) then project into lower dimension
                if len(subspace.shape) == 1:
                    if subspace.shape[0] > 50 :
                        extractors[new_key] = nn.Linear(subspace.shape[0], subspace.shape[0] //4)
                        total_concat_size += subspace.shape[0] //4
                    else: 
                        extractors[new_key] = nn.Identity() # ensure that batch dim gets added 
                        total_concat_size += gym.spaces.utils.flatdim(subspace)

                # scalars and not so long vectors get passed through
                elif len(subspace.shape) < 1:
                    extractors[new_key] = Add_Batch() # ensure that batch dim gets added 
                    total_concat_size += gym.spaces.utils.flatdim(subspace)

                # this is for the rotation_marix since it is the only field which has 2d shape but is not an image
                else:
                    extractors[new_key] = nn.Flatten(start_dim=1)
                    total_concat_size += gym.spaces.utils.flatdim(subspace)

        # Update the features dim manually
        self._features_dim = total_concat_size
        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            tensor = extractor(observations[self.key_mappings[key]])
            encoded_tensor_list.append(tensor)
            #print(key,observations[self.key_mappings[key]].shape)
            #print(key,tensor.shape)
         # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)


# TODO as for now this works but is not pretty. Thinking about refactoring it
class Add_Batch(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self,x):
        if len(x.shape) == 0: return x.unsqueeze(0).unsqueeze(0)
        if len(x.shape) ==1 : return x.unsqueeze(1)
        return x