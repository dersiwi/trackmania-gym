from trackmania_env.plotting.core import EnvPlotter
import cv2
import numpy as np
class Plot_Obs_Images(EnvPlotter):
    cmaps = {
        "rgb": cv2.COLOR_RGB2BGR,
        "grayscale": cv2.COLOR_RGB2GRAY
    }

    def __init__(self, img_size: tuple[int, int], color_space: str) -> None:
        super().__init__()
        
        if color_space not in self.cmaps:
            raise ValueError(f"Invalid color_space '{color_space}'. Must be one of: {list(self.cmaps.keys())}")
        self.cmap = self.cmaps[color_space]
        self.window_name = "Trackmania Obs"

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def setup_plot(self):
        return super().setup_plot()

    def plot(self, img: np.ndarray):

        print(img)
        plot_img = img.squeeze()

        if plot_img.ndim == 3 and plot_img.shape[0] in [1, 3, 4]:
            plot_img = plot_img.transpose(1, 2, 0) 
        # If image is float (e.g., 0.0 to 1.0), convert to uint8 (0 to 255)
        if plot_img.dtype != np.uint8:
            if np.max(plot_img) <= 1.0 and np.min(plot_img) >= 0.0:
                plot_img = (plot_img * 255).astype(np.uint8)
            else:
                # Assuming it's already float, just cast
                plot_img = plot_img.astype(np.uint8)
        print("casting success")
        plot_img = cv2.cvtColor(plot_img, self.cmap)
        print("cv2 color convert succes")
        cv2.imshow(self.window_name, plot_img)
        print("imshow succes")
        cv2.waitKey(1)
        print("wait key succes")
