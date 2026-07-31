import pyqtgraph as pg
#from trackmania_env.plotting.core import EnvPlotter
from PyQt5 import QtWidgets
import numpy as np
import sys


class Plot_Obs_Images:#(EnvPlotter):
    def __init__(self):
        self.app = pg.mkQApp("Image Observations")

        self.win = None
        self.img_view = None

    def setup_plot(self):
        self.win = QtWidgets.QMainWindow()
        self.win.resize(800, 600)

        self.img_view = pg.ImageView()
        self.img_view.setPredefinedGradient('viridis')

        self.win.setCentralWidget(self.img_view)
        self.win.show()

    def plot(self, image_tensor):
        img_np = image_tensor.squeeze().detach().cpu().numpy()

        if self.img_view is None:
            self.setup_plot()

        self.img_view.setImage(img_np, autoLevels=False, autoRange=False)
        self.app.processEvents()

    def close(self):
        if self.win:
            self.win.close()
            self.win = None
        self.img_view = None

if __name__ == "__main__":
    import torch, time
    plotter = Plot_Obs_Images()
    for i in range(200):
        img = torch.randn(3, 240, 320).abs()
        plotter.plot(img)
        time.sleep(0.03)
    plotter.close()
    sys.exit(plotter.app.exec_())
