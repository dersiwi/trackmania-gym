from trackmania_env.observations.observation_manager import ObservationManager
from gymnasium import spaces
import numpy as np
from configs.config import LinesightObsCfg
from tminterface.structs import SimStateData
import torch
from game_interaction.ipc_fields import IPCFields
import os
from PIL import Image
from torchvision.transforms.functional import to_pil_image


class ObservationTest(ObservationManager):

    def __init__(self, observation_list, colorspace, convert_torch, img_width, img_height, log_directory : str, log_frequency : int = 10):
        super().__init__(observation_list, colorspace, convert_torch, img_width, img_height)
        self.log_directory = log_directory
        self.log_frequency = log_frequency
        self.step = 0


    def get_observation(self, raw_observation : dict[str, np.ndarray | SimStateData]) -> np.ndarray | dict[str, np.ndarray] | torch.Tensor | dict[str, torch.Tensor]:
        """
        Takes raw observations from TMInterface and dissects them into image
        """
        self.step += 1

        processed_obs = super().get_observation(raw_observation)

        if self.step % self.log_frequency == 0:
            img = Image.fromarray(raw_observation[IPCFields.IMG])
            img.save(os.path.join(self.log_directory, f"raw_image_{self.step}.png"))

            if self.convert_torch:
                processed_img = processed_obs["image"].numpy()
            
            pimg = Image.fromarray(processed_img)
            pimg.save(os.path.join(self.log_directory, f"processed_image_{self.step}.png"))

        return processed_obs

        
        