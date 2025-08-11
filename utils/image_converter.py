import numpy as np
import torch
from PIL import Image

class ImageConverter:

    @staticmethod
    def bgra_to_rgb(image_bgra : np.ndarray) -> np.ndarray:
        """Converts brga image to rgb image"""
        rgb = image_bgra[:, :, :3][:, :, ::-1].copy()  # (H, W, 3) in RGB order
        chw = np.transpose(rgb, (2, 1, 0))  # (H, W, C) -> (C, W, H)
        return chw
        #return image_bgra[:, :, :3][:, :, ::-1].copy() #copy necessary because otherwise it has negative stride (aka memory accessed backwards) which is not supported by pytorch.
    
    @staticmethod
    def bgra_to_graysacle(image_bgra : np.ndarray) -> np.ndarray:
        """Converts bgra image to grayscale. Reuturns image as 8-bit-integer."""
        b, g, r = image_bgra[:, :, 0], image_bgra[:, :, 1], image_bgra[:, :, 2]
        gray : np.ndarray = 0.114 * b + 0.587 * g + 0.299 * r
        gray = np.round(gray).clip(0, 255)
        gray = gray[np.newaxis, :, :]
        return gray
    

    @staticmethod
    def save_image(img : np.ndarray | torch.Tensor, filepath : str) -> None:
        """Saves an image to the given path"""
        if type(img) == torch.Tensor:
            img = img.numpy()
        if img.shape[0] == 1: # grayscale
            img = img.squeeze()

        pimg = Image.fromarray((img * 255).astype('uint8'))
        pimg.save(filepath)