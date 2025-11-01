from trackmania_env.plotting.core import EnvPlotter
import matplotlib.pyplot as plt
import numpy as np

class Plot_Obs_Images(EnvPlotter):
    """Fast real-time image plotter using Matplotlib blitting."""

    def __init__(self):
        plt.ion()  # enable interactive mode
        self.image_handle = None
        self.bg_cache = None
        self.fig = None
        self.ax = None

    def setup_plot(self):
      pass

    def plot(self, image_tensor):
        """Display or update an image from a torch tensor."""
        img_np = image_tensor.squeeze().detach().cpu().numpy()

        # Handle grayscale or RGB
        if img_np.ndim == 2:
            display_img = img_np
            cmap = 'gray'
        elif img_np.ndim == 3 and img_np.shape[0] == 3:
            display_img = np.transpose(img_np, (1, 2, 0))  # (H, W, C)
            cmap = None
        else:
            raise ValueError(f"Unsupported image shape: {img_np.shape}")

        # First call → create figure and image
        if self.image_handle is None:
            self.fig, self.ax = plt.subplots(figsize=(4, 4))
            self.ax.axis('off')
            self.image_handle = self.ax.imshow(display_img, cmap=cmap, animated=True)
            self.fig.canvas.draw()
            self.bg_cache = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            plt.show(block=False)

        # Subsequent calls → update only the image (no full redraw)
        else:
            self.fig.canvas.restore_region(self.bg_cache)
            self.image_handle.set_data(display_img)
            self.ax.draw_artist(self.image_handle)
            self.fig.canvas.blit(self.ax.bbox)
            self.fig.canvas.flush_events()


