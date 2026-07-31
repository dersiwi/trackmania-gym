from trackmania_gym.plotting.plotter import EnvPlotter
import matplotlib.pyplot as plt
import numpy as np

class Plot_Obs_Images(EnvPlotter):
    """Fast real-time image plotter using Matplotlib blitting."""

    cmaps = {
            "rgb": None,
            "grayscale": "gray"
            }

    def __init__(self,img_size:tuple[int,int],color_space:str):

        if color_space not in self.cmaps:
            raise ValueError(f"Invalid color_space '{color_space}'. Must be one of: {list(self.cmaps.keys())}")
        cmap = self.cmaps[color_space]

        self.fig, self.ax = plt.subplots(figsize=(8, 6)) 
        fill_up = np.random.randn(*img_size)
        self.img = self.ax.imshow(fill_up,cmap= cmap)   
        self.ax.axis('off')

        self.fig.canvas.draw()
        self.bg_cache = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        plt.show(block=False)

    def setup_plot(self):
      pass

    def plot(self, img:np.ndarray):
        img = img.squeeze()
        if img.ndim == 3 and img.shape[0] == 3: img = img.transpose(2,1,0)
        self.img.set_data(img)
        self.ax.draw_artist(self.img)
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()


