"""Implementation of a custom feature extractor
https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html#multiple-inputs-and-dictionary-observations"""
import gymnasium as gym
import torch 
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TMN_Extractor(BaseFeaturesExtractor):
    """
    Combined feature extractor for the TrackMania environment observation space.
    This extractor is designed to work with observations containing  only vectors and image inputs. 
    It constructs a dedicated feature extractor for each key in the observation space

    All extracted features are then concatenated and passed through an optional combined 
    MLP (not shown here, but can be added after this module).

    :param observation_space: Gym Dict space describing the full observation structure.
    :param vision_model: A neural network used to extract features from image observations.
    :param out_dim: The number of dimension each extractor should project on to
    :param normalized_image: If True, assumes that image inputs are already normalized.
"""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        vision_model,
        out_dim:int = 64,
        device= "cpu",
        normalized_image: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim=1)

        total_concat_size = 0
        extractors: dict[str, nn.Module] = {}
        for key, subspace in observation_space.spaces.items():

            if key == "image":
                vision_model.to(device)
                extractors[key] = vision_model
                # check ouput dimension of vision model 
                dummy_input = (torch.zeros(1, *subspace.shape)).to(device) 
                dummy_output = vision_model(dummy_input)
                vision_model_out_dim = dummy_output.shape[1]
                assert vision_model_out_dim == out_dim
                total_concat_size += vision_model_out_dim
            else:
                hidden_dim = subspace.shape[0] // 2 if subspace.shape[0] > out_dim else subspace.shape[0] * 2

                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], hidden_dim, device=device),
                    nn.Linear(hidden_dim, out_dim, device=device)
                )
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