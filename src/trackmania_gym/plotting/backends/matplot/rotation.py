import numpy as np
import matplotlib.pyplot as plt
from plotting.plotter import EnvPlotter

class Rotation_Plotter(EnvPlotter):
    def __init__(self):
        super().__init__()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-1, 1])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Live Rotation Matrix Axes")

        self.ref_x, = self.ax.plot([0, 1], [0, 0], [0, 0], color='lightcoral', linestyle='--', label='X (ref)')
        self.ref_y, = self.ax.plot([0, 0], [0, 1], [0, 0], color='lightgreen', linestyle='--', label='Y (ref)')
        self.ref_z, = self.ax.plot([0, 0], [0, 0], [0, 1], color='lightblue', linestyle='--', label='Z (ref)')

        self.origin = np.array([0, 0, 0])
        self.x_line, = self.ax.plot([0, 1], [0, 0], [0, 0], color='r', label='X-axis',animated = True)
        self.y_line, = self.ax.plot([0, 0], [0, 1], [0, 0], color='g', label='Y-axis',animated = True)
        self.z_line, = self.ax.plot([0, 0], [0, 0], [0, 1], color='b', label='Z-axis',animated = True)
        self.ax.legend()

        # Save background for blitting
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        plt.show(block=False)

    def plot(self, rot_matrix):
        # Restore background
        self.fig.canvas.restore_region(self.background)

        x_axis = rot_matrix[:, 0]
        y_axis = rot_matrix[:, 1]
        z_axis = rot_matrix[:, 2]

        self.x_line.set_data_3d([0, x_axis[0]], [0, x_axis[1]], [0, x_axis[2]])
        self.y_line.set_data_3d([0, y_axis[0]], [0, y_axis[1]], [0, y_axis[2]])
        self.z_line.set_data_3d([0, z_axis[0]], [0, z_axis[1]], [0, z_axis[2]])

        self.ax.draw_artist(self.x_line)
        self.ax.draw_artist(self.y_line)
        self.ax.draw_artist(self.z_line)

        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()

    def setup_plot(self):
        return super().setup_plot()
