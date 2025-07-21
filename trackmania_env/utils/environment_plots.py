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
