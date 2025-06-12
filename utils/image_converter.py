import numpy as np

class ImageConverter:

    @staticmethod
    def bgra_to_rgb(image_bgra : np.ndarray) -> np.ndarray:
        """Converts brga image to rgb image"""
        return image_bgra[:, :, :3][:, :, ::-1].copy() #copy necessary because otherwise it has negative stride (aka memory accessed backwards) which is not supported by pytorch.
    
    @staticmethod
    def bgra_to_graysacle(image_bgra : np.ndarray) -> np.ndarray:
        """Converts bgra image to grayscale. Reuturns image as 8-bit-integer."""
        b, g, r = image_bgra[:, :, 0], image_bgra[:, :, 1], image_bgra[:, :, 2]
        gray = 0.114 * b + 0.587 * g + 0.299 * r
        return gray.astype(np.uint8)