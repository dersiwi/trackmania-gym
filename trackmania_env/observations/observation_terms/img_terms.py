from abc import ABC,abstractmethod
import numpy as np

from gymnasium.spaces import Box

from trackmania_env.observations.observation_term import ObservationTerm
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
        #Converts bgra image to grayscale
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray : np.ndarray = 0.114 * b + 0.587 * g + 0.299 * r
        gray = gray[np.newaxis, :, :]
        return gray

class RGBImgConverter(ImgConverter):
    def __init__(self):
        super().__init__(num_channels= 3)

    def cnvt_img(self, img: np.ndarray) -> np.ndarray:
        #Converts brga image to rgb image
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

    def __init__(self,normalize=False, name="image",colorspace=Colorspace.GRAYSCALE,img_width:int = 128, img_height:int = 128, dtype=np.uint8):
        super().__init__(name, normalize)

        try:
            Img_Converter_Class = self._COLORSPACE_TO_CLASS[colorspace]
        except KeyError:
            raise ValueError(f"Unsupported colorspace: {colorspace}")

        self.dtype = dtype
        self.img_converter: ImgConverter = Img_Converter_Class()
        self.observation_space = Box(
            low=0,
            high= 1 if normalize else 255,
            shape=(self.img_converter.num_channels,img_height,img_width),
            dtype=self.dtype)

    def _get_obs(self, game_states, **kwargs):
        #this will be a bgra img and already has the shape of img_height x img_width.
        # NOTE maybe think of doing the rescaling of the img here but i am not sure sicne the img rescaling does TMI on its own
        img = game_states[IPCFields.IMG] 
        img = self.img_converter.cnvt_img(img)
        return img.astype(dtype=self.dtype)

    def _normalize(self, obs):
        # obs will be an image
        return (obs / 255).astype(self.dtype)
