from abc import ABC,abstractmethod
import numpy as np

from trackmania_env.observations.observation_manager import ObservationTerm
from game_interaction.ipc_fields import IPCFields


class ImgConverter(ABC):
    def __init__(self, num_channels: int):
        self._num_channels = num_channels

    @abstractmethod
    def cnvt_img(self, img: np.ndarray) -> np.ndarray:
        # the images tmnf interface returns are per default rgba
        raise NotImplementedError

    @property
    def num_channels(self):
        return self._num_channels

class GrayScaleImgConverter(ImgConverter):
    def __init__(self):
        super().__init__(num_channels= 1)

    def cnvt_img(self, img: np.ndarray) -> np.ndarray:
        """Converts bgra image to grayscale. Returns image as 8-bit-integer."""
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray : np.ndarray = 0.114 * b + 0.587 * g + 0.299 * r
        gray = gray[np.newaxis, :, :]
        return gray

class RGBImgConverter(ImgConverter):
    def __init__(self):
        super().__init__(num_channels= 3)

    def cnvt_img(self, img: np.ndarray) -> np.ndarray:
        """Converts brga image to rgb image"""
        rgb = img[:, :, :3][:, :, ::-1].copy()  # (H, W, 3) in RGB order
        chw = np.transpose(rgb, (2, 1, 0))  # (H, W, C) -> (C, W, H)
        return chw

class BGRAImgConverter(ImgConverter):
    def __init__(self):
        super().__init__(num_channels= 4)

    def cnvt_img(self, img: np.ndarray) -> np.ndarray:
        return img
    
class ImageObservationTerm(ObservationTerm):
    class Colorspace:
        GRAYSCALE = "grayscale"
        RGB = "rgb"
        BGRA = "bgra"

    # Mapping colorspaces to classes
    _COLORSPACE_TO_CLASS = {
        Colorspace.GRAYSCALE: GrayScaleImgConverter,
        Colorspace.RGB: RGBImgConverter,
        Colorspace.BGRA: BGRAImgConverter,
    }

    def __init__(self,convert_torch, normalize, name="image",colorspace=Colorspace.GRAYSCALE, dtype=np.uint8):
        super().__init__(name,convert_torch, normalize)
        self.dtype = dtype

        try:
            Img_Converter_Class = self._COLORSPACE_TO_CLASS[colorspace]
        except KeyError:
            raise ValueError(f"Unsupported colorspace: {colorspace}")

        self.img_converter: ImgConverter = Img_Converter_Class(convert_torch, normalize)

    def _get_obs(self, game_states, **kwargs):
        img = [IPCFields.IMG] #this will be a bgra img
        img = self.img_converter.cnvt_img(img)
        return img.astype(dtype=self.dtype)

    def _normalize(self, obs):
        # obs will be an image
        obs = obs/255
        return obs.astype(dtype=self.dtype)

    def get_observation_space(self):
        return self.img_obs_term.get_observation_space()