from abc import ABC, abstractmethod
import numpy as np
import os
from scipy.stats import norm

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

class EnvPlotter(ABC):
    """
    Abstract base class for environment plotters.
    Subclasses must implement setup_plot() and plot() methods.
    """

    @abstractmethod
    def setup_plot(self):
        pass

    @abstractmethod
    def plot(self):
        pass


class Plot_Obs_Images(EnvPlotter):
    """
    This plots the images which are part of the observation space in realtime
    """
    def __init__(self):
        self.fig = None
        self.ax = None
        self.image_handle = None
        #self.setup_plot()

    def setup_plot(self):
        """
        Set up the figure and axis for displaying image observations.
        """
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.axis('off')
        plt.ion()
        plt.show()

    def plot(self, image_tensor):
        """
        Plot or update the image from a PyTorch tensor.
        Expects image_tensor of shape (1, C, H, W) or (1, H, W) where C is either 1 or 3.
        """
        img_np = image_tensor.squeeze().cpu().numpy()  # (H, W)

        # Handle grayscale (H, W)
        if img_np.ndim == 2:
            display_img = img_np
            cmap = 'gray'
        # Handle RGB (3, H, W) -> (H, W, 3)
        elif img_np.ndim == 3 and img_np.shape[0] == 3:
            display_img = img_np.transpose(2, 1, 0)
            cmap = None
        else:
            raise ValueError(f"Unsupported image shape: {img_np.shape}")

        if self.image_handle is None:
            self.image_handle = self.ax.imshow(display_img, cmap=cmap)
            self.ax.axis('off')
        else:
            self.image_handle.set_data(display_img)

        self.fig.canvas.draw()
        plt.pause(0.001)


class Plot_Rewards(EnvPlotter):
    """
    This plots all the reward terms which get used in the reward manager.
    """
    def __init__(self, key_to_plot = None, y_lim=(-1, 1)):
        super().__init__()
        self.fig = None
        self.ax = None
        self.key_to_plot = key_to_plot
        self.y_lim = y_lim 
        self.vals = {}
        self.lines = {}

    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(13, 13))
        plt.ion()
        plt.show()

    def plot(self, rewards):
        for k in rewards:
            if k not in self.vals: self.vals[k] = []
            self.vals[k].append(rewards[k])

        if self.key_to_plot is None:
            self.ax.set_title("Tracking rewards over time")
            self.ax.set_ylabel("Reward Values per env step")

            for key, values in self.vals.items():
                if key not in self.lines:
                    line, = self.ax.plot(values, label=key)
                    self.lines[key] = line
                else:
                    self.lines[key].set_ydata(values)
                    self.lines[key].set_xdata(range(len(values)))

        else:
            self.ax.set_title(f"Tracking '{self.key_to_plot}' Over Time")
            self.ax.set_ylabel(self.key_to_plot)
            values = self.vals[self.key_to_plot]
            if self.key_to_plot not in self.lines:
                line, = self.ax.plot(values, label=self.key_to_plot)
                self.lines[self.key_to_plot] = line
            else:
                self.lines[self.key_to_plot].set_ydata(values)
                self.lines[self.key_to_plot].set_xdata(range(len(values)))

        self.ax.set_xlabel("Step")
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.legend()
        plt.draw()
        plt.pause(0.001)

class Plot_Lateral_Distance(EnvPlotter):
    def __init__(self, reference_line_manager):
        super().__init__()
        self.ref_line_manager =  reference_line_manager
        self.reference_line = self.ref_line_manager.reference_line

        self.fig, self.ax = None,None
        self.bar_fig, self.bar_ax = None,None
        self.gauss_plot = None 

        #TODO maybe use the correct values from the nextpoint rewards obs manager gaussian
        self.gauss_dist = norm(0,np.sqrt(12))
        # Create values for the smooth Gaussian curve
        self.x_vals = np.linspace(-50, 50, 500)
        self.y_vals = self.gauss_dist.pdf(self.x_vals)
        self.gauss_curve = self.gauss_dist.rvs(size=1000) 

        #self.setup_plot()

        plt.ion()
        plt.show()
    
    def setup_plot(self):
        # Create ONE figure
        self.fig = plt.figure(figsize=(12, 8))

        # Use GridSpec to control layout
        gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])

        # Left side: map view takes two rows
        self.ax_map = self.fig.add_subplot(gs[:, 0])
        self.ax_map.set_title("XZ View (World Space)")
        self.ax_map.set_xlabel("X Axis")
        self.ax_map.set_ylabel("Z Axis")
        self.ax_map.set_aspect('equal')

        # Top-right: bar plot
        self.ax_bar = self.fig.add_subplot(gs[0, 1])
        self.bar_container = self.ax_bar.bar(["Distance"], [0.0], color='magenta')
        self.ax_bar.set_ylim(0, 10)
        self.ax_bar.set_ylabel("Euclidean Distance")
        self.ax_bar.set_title("Lateral Distance")

        # Bottom-right: Gaussian PDF plot
        self.ax_gauss = self.fig.add_subplot(gs[1, 1])
        self.ax_gauss.plot(self.x_vals, self.y_vals, label='Gaussian PDF', color='blue')
        self.ax_gauss.set_title("Lateral Distance vs Gaussian PDF")
        self.ax_gauss.set_xlabel("Lateral Distance")
        self.ax_gauss.set_ylabel("Probability Density")
        self.ax_gauss.legend()

        self.fig.tight_layout()
    
    def plot(self, info):
        position = info["position"]
        next_refline_index = info["next_refline_index"]
        ref_line_point = self.reference_line[next_refline_index]
        diff_vec = ref_line_point - position
        lateral_distance = self.ref_line_manager.calculate_lateral_difference(next_refline_index, position)

        # === Map View ===
        self.ax_map.cla()
        self.ax_map.set_title("XZ View (World Space)")
        self.ax_map.set_xlabel("X Axis")
        self.ax_map.set_ylabel("Z Axis")
        self.ax_map.set_aspect('equal')
        self.ax_map.plot(-1. * self.reference_line[:, 0], self.reference_line[:, 2], color='black')
        self.ax_map.scatter(-1. * position[0], position[2], color='green', marker='x', s=50)
        self.ax_map.scatter(-1. * ref_line_point[0], ref_line_point[2], color='blue', marker='x', s=50)
        self.ax_map.arrow(-1. * position[0], position[2], -1. * diff_vec[0], diff_vec[2],
                          head_width=0.1, head_length=0.2, fc='magenta', ec='magenta', length_includes_head=True)

        # === Bar Plot ===
        self.ax_bar.cla()
        self.bar_container = self.ax_bar.bar(["Distance"], [lateral_distance], color='magenta')
        self.ax_bar.set_ylim(0, max(10, lateral_distance + 1))
        self.ax_bar.set_title(f"Distance to Ref Point: {lateral_distance:.2f}")
        self.ax_bar.set_ylabel("Euclidean Distance")

        # === Gaussian Plot ===
        self.ax_gauss.cla()
        prob = self.gauss_dist.pdf(lateral_distance)
        self.ax_gauss.plot(self.x_vals, self.y_vals, label='Gaussian PDF', color='blue')
        self.ax_gauss.axvline(lateral_distance, color='red', linestyle='--', label=f'x = {lateral_distance:.2f}')
        self.ax_gauss.plot(lateral_distance, prob, 'ro')
        self.ax_gauss.set_title("Lateral Distance vs Gaussian PDF")
        self.ax_gauss.set_xlabel("Lateral Distance")
        self.ax_gauss.set_ylabel("Probability Density")
        self.ax_gauss.legend()

        self.fig.tight_layout()
        self.fig.canvas.draw()
        plt.pause(0.001)

class Plot_RefLine(EnvPlotter):
    def __init__(self, reference_line : np.ndarray):
        super().__init__()
        self.reference_line = reference_line
        self.fig, self.ax = None,None
        plt.ion()
        plt.show()

    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(11, 11))
        self.ax.set_title("XZ View (World Space)")
        self.ax.set_xlabel("X Axis")
        self.ax.set_ylabel("Z Axis")
        self.ax.set_aspect('equal')

    def plot(self,info):
        
        points = info["comming_refline_points"]
        orientation = info["orientation"]
        inv_orientation = orientation.T
        position = info["position"]
        next_refline_index = info["next_refline_index"]

        # 1. Check determinant ≈ 1 since in_rotation should be a rotation matrix 
        det = np.linalg.det(inv_orientation)
        assert np.isclose(det, 1.0, atol=1e-5), f"Determinant not close to 1: det = {det}"
        # 2. Check if inv_orientation @ orientation ≈ Identity
        ident = inv_orientation @ orientation
        assert np.allclose(ident, np.eye(3), atol=1e-5), f"Matrix product not identity:\n{ident}"

        # Transform points to world coordinates
        points = (inv_orientation @ points.T).T + position

        # Clear and redraw plot
        self.ax.cla()

        # Plot reference line (black), transformed points (red), and position (green), and next reference-line-point (blue)
        self.ax.plot(-1. * self.reference_line[:, 0], self.reference_line[:, 2], color='black', linestyle='-')
        self.ax.plot(-1. * points[:, 0], points[:, 2], color='red', marker='o', linestyle='-')
        self.ax.scatter(-1. * position[0], position[2], color='green', marker='x', s=50)
        self.ax.scatter(-1. * self.reference_line[next_refline_index, 0], self.reference_line[next_refline_index, 2], color='blue', marker='x', s=50)
        
        self.fig.canvas.draw()
        plt.pause(0.001)

class PrintRotation(EnvPlotter):
    def __init__(self):
        super().__init__()
        self.quiver = None
        self.fig, self.ax = None,None
        plt.ion()
        plt.show()

    def setup_plot(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def plot(self, rot_matrix):
        # Clear and redraw
        self.ax.cla()
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-1, 1])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Live Rotation Matrix Axes")
        # Initial dummy arrows
        self.quiver = self.ax.quiver(0, 0, 0, 1, 0, 0, color='r', label="X-axis")
        self.quiver = self.ax.quiver(0, 0, 0, 0, 1, 0, color='g', label="Y-axis")
        self.quiver = self.ax.quiver(0, 0, 0, 0, 0, 1, color='b', label="Z-axis")
        self.ax.legend()

        origin = np.array([0, 0, 0])
        x_axis = rot_matrix[:, 0]
        y_axis = rot_matrix[:, 1]
        z_axis = rot_matrix[:, 2]

        self.ax.quiver(*origin, *x_axis, color='r', label="X-axis")
        self.ax.quiver(*origin, *y_axis, color='g', label="Y-axis")
        self.ax.quiver(*origin, *z_axis, color='b', label="Z-axis")
        plt.draw()
        plt.pause(0.001)

class Plot_1D_Values(EnvPlotter):
    """
    This plots the 1D values specified by keys_to_plot from the environment.
    """
    def __init__(self, keys_to_plot, y_lim = None):
        """
        Initialize the callback to plot multiple keys over time.

        Args:
            keys_to_plot (list of str): List of keys from the `info` dict to plot.
            y_lim (tuple): Y-axis limits for the plot.
        """
        super().__init__()
        self.keys_to_plot = keys_to_plot if isinstance(keys_to_plot, list) else [keys_to_plot]
        self.y_lim = y_lim
        self.vals = {key: [] for key in self.keys_to_plot}

        self.fig, self.ax = None,None
        plt.ion()
        plt.show()
  
    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        
    def plot(self,data):
        
        # Clear and redraw the plot
        self.ax.cla()
        self.ax.set_title("Debug Plot")
        self.ax.set_xlabel("Timestep")
        if self.y_lim: self.ax.set_ylim(*self.y_lim)

        for key in self.keys_to_plot:
            self.vals[key].append(data[key])
            self.ax.plot(range(len(self.vals[key])), self.vals[key], label=key)

        self.ax.legend()
        plt.draw()
        plt.pause(0.001)


class Plot_3D_Values(EnvPlotter):
    """
    This plots 3D vector values (x, y, z) over time from the environment info dict.
    """
    def __init__(self, key_to_plot, y_lim= None):
        """
        Initialize the callback to plot the 3D vector values over time.

        Args:
            key_to_plot (str): Key from the `info` dict containing a 3D vector.
            y_lim (tuple): Y-axis limits for the plots.
        """
        super().__init__()
        self.key_to_plot = key_to_plot
        self.y_lim = y_lim
        self.vals = []

        self.fig, self.axes = None, None
        plt.ion()
        plt.show()

    def setup_plot(self):
        self.fig, self.axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)

    def plot(self, data):
        val = np.asarray(data[self.key_to_plot])
        self.vals.append(val)

        # Clear and redraw the subplots
        for ax in self.axes:
            ax.cla()

        vals_array = np.array(self.vals)  # shape: (timesteps, 3)
        labels = ['x', 'y', 'z']
        colors = ['r', 'g', 'b']

        for i in range(3):
            self.axes[i].plot(range(len(vals_array)), vals_array[:, i], color=colors[i], label=labels[i])
            self.axes[i].set_ylabel(labels[i])
            if self.y_lim: self.axes[i].set_ylim(*self.y_lim)
            self.axes[i].legend()
            self.axes[i].grid(True)

        self.axes[2].set_xlabel("Timestep")
        self.fig.suptitle(f"3D Vector Plot: '{self.key_to_plot}'")

        plt.draw()
        plt.pause(0.001)