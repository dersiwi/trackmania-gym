import numpy as np
import torch

from gymnasium import spaces
from tminterface.structs import SimStateData
from game_interaction.ipc_fields import IPCFields
from utils.image_converter import ImageConverter


class ObservationManager:

    class Colorspace:
        GRAYSCALE = 0
        RGB = 1
        BGRA = 2

        REV_DICT = {"grayscale" : 0, "rgb" : 1, "rgba" : 2} #this is somewhat ugly but this way the config contains a readable string

    def __init__(self, colorspace : str, convert_torch : bool, img_width : int, img_height : int, obs_have_img: bool = True):
        self.obs_have_img = obs_have_img
        self.colorspace : int = ObservationManager.Colorspace.REV_DICT[colorspace]
        self.convert_torch : bool = convert_torch
        self.img_width = img_width
        self.img_height = img_height

        self.env = None
        self.n_channels = 1 if self.colorspace == ObservationManager.Colorspace.GRAYSCALE else 3

        self.info = {}


    def set_env(self, environment):
        from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
        self.env : TMNF_Single_Agent_Env = environment


    def get_values_from_state_dict(self, obs : SimStateData) -> np.ndarray:
        """This method gets a raw-observation simstate-obejcet (@param : obs) and converts it into a flattened
        vector, that is given to the feature-extractor network of the policy.
        
        Returns
        -------
        Vector of shape [N,] where N is the amount of non-image observation-fields."""
        raise NotImplementedError("Do not use this method directly, use on of the implementations of this method.")

    def cnvt_imgs(self, images : np.ndarray) -> np.ndarray | torch.Tensor:
        """Converts image given by simulation into specified colortype and normalizes them into [0,1]."""
        if self.colorspace == ObservationManager.Colorspace.RGB:
            imgs = ImageConverter.bgra_to_rgb(images)
        elif self.colorspace == ObservationManager.Colorspace.GRAYSCALE:
            imgs = ImageConverter.bgra_to_graysacle(images)
        
        if self.convert_torch:
            imgs = torch.from_numpy(imgs)

        return imgs / 255

    def get_observation_dict(self) -> spaces.Dict:
        """Returns observation dict for environment according to initialization.
        In most cases this is going to be a dictonray-observation space containing one 'image' field and one 'state'-field."""
        raise NotImplementedError("Do not use this method directly, use on of the implementations of this method.")
        
        
    def reset(self):
        """Resets the observation-manager. This is called if the environment is reset."""
        pass
    
    def get_observation(self, raw_observation : dict[str, np.ndarray | SimStateData]) -> tuple[np.ndarray | dict[str, np.ndarray] | torch.Tensor | dict[str, torch.Tensor],dict[str,any]]:
        """
        Takes raw observations from TMInterface and dissects them into image
        """
        state_observation_vector = self.get_values_from_state_dict(raw_observation[IPCFields.SIMSTATE])
        if self.convert_torch:
            state_observation_vector = torch.from_numpy(state_observation_vector)


        if self.obs_have_img:
            imgs = self.cnvt_imgs(raw_observation[IPCFields.IMG])
            assert imgs.shape == (self.n_channels, self.img_height, self.img_width), f"Expected shape to be ({self.n_channels},{self.img_height}, {self.img_width}) but got {imgs.shape}"
            return {"image" : imgs, "state" : state_observation_vector},self.info
        else:
            return state_observation_vector,self.info