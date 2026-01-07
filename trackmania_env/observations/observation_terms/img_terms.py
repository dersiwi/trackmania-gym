from __future__ import annotations
import numpy as np
import torch
import os

from PIL import Image
from gymnasium.spaces import Box

from trackmania_env.observations.observation_term import ObservationTerm
from game_interaction.ipc_fields import IPCFields


class ImgConverter:
    GRAYSCALE = "grayscale"
    RGB = "rgb"
    BGRA = "bgra"
    channel_map = {GRAYSCALE: 1,RGB: 3, BGRA: 4}

    def save_image(self, img : np.ndarray | torch.Tensor, filepath : str) -> None:
        """Saves an image to the given path"""
        if type(img) == torch.Tensor:
            img = img.numpy()
        if img.shape[0] == 1: # grayscale
            img = img.squeeze()

        pimg = Image.fromarray((img * 255).astype('uint8'))
        pimg.save(filepath)

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

        self.dump_freq = -1 # only if greater 0 images are dumped
        self.dump_dir = None
        self.n_imgs = 0
        self.imgs_to_dump = 0


    def set_dump_freq(self, freq : int, dirpath : str) -> None:
        """Sets a dumping frequency, to dump converted images, aka. the observations.
        Args:
            freq (int)      : Every freq-environment steps, the 5 consecutive images are dumped.
            dirpath (str)  : Path to directory where images are going to be dumped"""
        self.dump_freq = freq
        self.dump_dir = dirpath
        assert freq > 5, "Cannot go lower than 5 due to magic numbers"


    def _get_obs(self, game_states, **kwargs):
        #this will be a bgra img and already has the shape of img_height x img_width.
        # NOTE maybe think of doing the rescaling of the img here but i am not sure sicne the img rescaling does TMI on its own
        img = game_states[IPCFields.IMG] 
        img = self.img_converter.cnvt_img(img)
        # NOTE: when we are sure this work we will removw the assert
        assert img.shape[0] == self.num_channels , f"Expected {self.num_channels} color channels, got {img.shape[0]}"
        
        # Dump images if active 
        if self.imgs_to_dump > 0 or (self.dump_freq > 0 and self.n_imgs % self.dump_freq == 0):
            self.img_converter.save_image(img, filepath=os.path.join(self.dump_dir, f"control_img_{self.n_imgs}.png"))
            self.imgs_to_dump = 5 if self.n_imgs % self.dump_freq == 0 else self.imgs_to_dump
            self.imgs_to_dump -= 1

        self.n_imgs += 1

        return img.astype(dtype=self.dtype), {}

    def _normalize(self, obs):
        return (obs / 255).astype(self.dtype)
    

    def flatten(self, processed_obs):
        return processed_obs.reshape(processed_obs.shape[0] * processed_obs.shape[1] * processed_obs.shape[2])
    
    def get_flatten_dim(self):
        return self.num_channels * self.img_height * self.img_width
    
    def get_native_shape(self):
        return (self.num_channels , self.img_height , self.img_width)
