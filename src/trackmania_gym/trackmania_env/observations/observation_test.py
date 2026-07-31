import numpy as np
import torch

import os
from PIL import Image
from tminterface.structs import SimStateData

from trackmania_gym.game_interaction.ipc_fields import IPCFields
from trackmania_gym.trackmania_env.observations.observation_manager import ObservationManager


class ObservationTest(ObservationManager):

    def __init__(self, obs_mangager : ObservationManager, colorspace, convert_torch, img_width, img_height, log_directory : str, normalize_obs : bool = False, log_frequency : int = 10, log_images : bool = True, max_logs = 50):
        super().__init__(colorspace, convert_torch, img_width, img_height, normalize_obs=normalize_obs)
        self.log_directory = log_directory
        self.log_frequency = log_frequency
        self.step = 0
        self.log_images = True
        self.obs_manager = obs_mangager
        self.max_logs = max_logs
        os.makedirs(self.log_directory, exist_ok=True)

    def set_env(self, environment):
        self.obs_manager.set_env(environment)
        return super().set_env(environment)
    
    def get_observation_dict(self):
        return self.obs_manager.get_observation_dict()


    def get_observation(self, raw_observation : dict[str, np.ndarray | SimStateData]) -> np.ndarray | dict[str, np.ndarray] | torch.Tensor | dict[str, torch.Tensor]:
        """
        Takes raw observations from TMInterface and dissects them into image
        """
        self.step += 1

        processed_obs, info = self.obs_manager.get_observation(raw_observation)

        if self.step % self.log_frequency == 0 and self.log_images and self.step < self.max_logs:
            assert raw_observation[IPCFields.IMG].shape == (self.img_width, self.img_height, 4), f"Got unexpected shape {raw_observation[IPCFields.IMG].shape}"
            
            np.save(os.path.join(self.log_directory, f"raw_image_{self.step}.npy"), raw_observation[IPCFields.IMG])
            img = Image.fromarray(raw_observation[IPCFields.IMG].astype('uint8'), 'RGBA')
            img.save(os.path.join(self.log_directory, f"raw_image_{self.step}.png"))

            if self.convert_torch:
                processed_img = processed_obs["image"].numpy()
                np.save( os.path.join(self.log_directory,f"processed_image_{self.step}.npy"), processed_img)
            if self.colorspace == ObservationManager.Colorspace.GRAYSCALE:
                processed_img = processed_img.squeeze()
                assert processed_img.shape == (self.img_height, self.img_width), f"Got unexpected shape {processed_img.shape}"

            pimg = Image.fromarray((processed_img * 255).astype('uint8'))
            pimg.save(os.path.join(self.log_directory, f"processed_image_{self.step}.png"))

            # print position of and veloctiy of car.
            ssD : SimStateData = raw_observation[IPCFields.SIMSTATE]
            print(ssD.position, ssD.velocity)

        return processed_obs, info