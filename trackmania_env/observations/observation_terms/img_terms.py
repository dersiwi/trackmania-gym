from __future__ import annotations
import numpy as np

from gymnasium.spaces import Box

from trackmania_env.observations.observation_term import ObservationTerm
from game_interaction.ipc_fields import IPCFields


class ImgConverter:
    GRAYSCALE = "grayscale"
    RGB = "rgb"
    BGRA = "bgra"
    channel_map = {GRAYSCALE: 1,RGB: 3, BGRA: 4}

    def __init__(self, colorspace : str):
        self._num_channels = ImgConverter.channel_map[colorspace]
        self.colorspace = colorspace
        assert self.colorspace in ImgConverter.channel_map.keys(), f"Given colorspace '{colorspace}' did not match any implemented colorspaces." 

    def cnvt_img(self, img: np.ndarray) -> np.ndarray:
        """ the images tmnf interface returns are per default rgba """
        if self.colorspace == ImgConverter.GRAYSCALE: 
            b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            gray : np.ndarray = 0.114 * b + 0.587 * g + 0.299 * r
            gray = gray[np.newaxis, :, :]
            return gray
        
        elif self.colorspace == ImgConverter.RGB:
            #Converts brga image to rgb image
            rgb = img[:, :, :3][:, :, ::-1].copy()  # (H, W, 3) in RGB order
            chw = np.transpose(rgb, (2, 1, 0))  # (H, W, C) -> (C, W, H)
            return chw
        else:
            return img  #is already in RGBA format

    @property
    def num_channels(self):
        return self._num_channels
    
class ImageObservationTerm(ObservationTerm):

    def __init__(self,normalize = True, name="image", colorspace="grayscale", img_width : int = 128, img_height : int = 128, dtype=np.float32):
        super().__init__(name, normalize)

        
        self.dtype = dtype
        self.img_converter = ImgConverter(colorspace)
        self.num_channels = self.img_converter.num_channels

        self.img_width = img_width
        self.img_height = img_height

        self.observation_space = Box(
            low=0,
            high= 1 if normalize else 255, 
            shape=(self.img_converter.num_channels,img_height,img_width), 
            dtype=self.dtype)
        
        if self.normalize and np.issubdtype(self.dtype, np.integer):
            print(f"Storing normalized images as uint8 will lead to data loss, Either disable normalization or set dtype to float32. \nCurrent dtype : {self.dtype}")


    def _get_obs(self, game_states, **kwargs):
        #this will be a bgra img and already has the shape of img_height x img_width.
        # NOTE maybe think of doing the rescaling of the img here but i am not sure sicne the img rescaling does TMI on its own
        img = game_states[IPCFields.IMG] 
        img = self.img_converter.cnvt_img(img)
        # NOTE: when we are sure this work we will removw the assert
        assert img.shape[0] == self.num_channels , f"Expected {self.num_channels} color channels, got {img.shape[0]}"

        return img.astype(dtype=self.dtype), {}

    def _normalize(self, obs):
        return (obs / 255).astype(self.dtype)
    

    def flatten(self, processed_obs):
        return processed_obs.reshape(processed_obs[0] * processed_obs[1] * processed_obs[2])
    
    def get_flatten_dim(self):
        return self.num_channels * self.img_height * self.img_width
    
    def get_native_shape(self):
        return (self.num_channels , self.img_height , self.img_width)
