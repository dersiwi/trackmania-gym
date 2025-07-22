"""
Module implements several test cases (aka. TestEnvironmentCallback's) to test the return values and behaviour of the environment.
Implemented TestCases;

## Basic Test Cases

    - PrintRewardsToConsole : Prints rewards to the console
    - TrackVelocity         : Tracks x,y,z velocity over time and displays them after testing

## RealTime-Visualizations (Live while driving in game)

    - PrintRotation         : Prints global coordinate system as well as car-coordinate system to visualize current rotation of the car (3d)
    - PrintVectorToNextReferencePoint   : 3d Visualization of vector of car to next-reference-line-point
    - PrintVector2DToNextReferencePoint : 2d Visualization of vector of car to next-reference-line-point (xz-plane)
    - Test_1D_Next_Point_Manager        : Visualizes several info-values given in each environment step in real time, TODO : which ones?
    - Test_3D_Next_Point_Manager        : Visualizes several info-values given in each environment step in real time, TODO : which ones?
    - Test_Lateral_Dist_Next_Point_Manager : Tetss and prints lateral distance
    - Test_Reward_Next_Point_Manager    : Prints specific, or all rewards in realtime
    - PretrainingDataCollection         : Used to collect images as well as lateral distances to create training set for pretraining Vision Encoder.
"""
import numpy as np
import os
from scipy.stats import norm
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from trackmania_env.envs.testenv_single_agent import TestEnvironmentCallback, Live3dPlotEnvironmentCallback
from trackmania_env.utils.reference_line_manager import ReferenceLineManager

class PrintRewardsToConsole(TestEnvironmentCallback):

    def __init__(self):
        super().__init__()

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        if self.n_step % 128 == 0:
            for key in info["rewards"]:
                print(key, end=" | ")
            print("\n")
        for key in info["rewards"]:
            print(key,info["rewards"][key], end=" | ")
        print("\n")
        self.n_step += 1

class TrackVelocity(TestEnvironmentCallback):
    """Tacks vx, vy, vz and plots them after run."""
    
    def __init__(self):
        super().__init__()
        self.velocities = [[], [], []] #v_x, v_y, v_z

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        velocity = info["velocity"]
        for i in range(len(self.velocities)):
            self.velocities[i].append(velocity[i])


    def _call_after_run(self):
        time = range(len(self.v_x))  

        plt.figure(figsize=(12, 6))

        for idx, name, color in zip([1,2,3], ["v_x", "v_y", "v_z"], ["blue", "orange", "green"]):

            plt.subplot(3, 1, idx)
            plt.plot(time, self.velocities[idx-1], label=name, color=color)
            plt.ylabel(name)
            if idx == 3:
                plt.xlabel('Time Step')
            plt.grid(True)

        plt.tight_layout()
        plt.show()

class PrintRotation(Live3dPlotEnvironmentCallback):
    def __init__(self):
        self.quiver = None
        super().__init__()

    def _setup_plot(self):
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

    def _update_plot(self, rot_matrix):
        # Clear and redraw
        self.ax.cla()
        self._setup_plot()

        origin = np.array([0, 0, 0])
        x_axis = rot_matrix[:, 0]
        y_axis = rot_matrix[:, 1]
        z_axis = rot_matrix[:, 2]

        self.ax.quiver(*origin, *x_axis, color='r', label="X-axis")
        self.ax.quiver(*origin, *y_axis, color='g', label="Y-axis")
        self.ax.quiver(*origin, *z_axis, color='b', label="Z-axis")
        plt.draw()
        plt.pause(0.001)

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        rot_matrix = np.array(info["rotation_matrix"])
        self._update_plot(rot_matrix)


class PrintVectorToNextReferencePoint(Live3dPlotEnvironmentCallback):
    def __init__(self):
        self.quiver = None
        super().__init__()



    def _setup_plot(self):
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-2, 2])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Live Rotation Matrix Axes")
        # Initial dummy arrows
        self.quiver = self.ax.quiver(0, 0, 0, 1, 0, 0, color='r', label="X-axis")
        self.quiver = self.ax.quiver(0, 0, 0, 0, 1, 0, color='g', label="Y-axis")
        self.quiver = self.ax.quiver(0, 0, 0, 0, 0, 1, color='b', label="Z-axis")
        self.ax.legend()



    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        
        self.ax.cla()
        self._setup_plot()

        origin = np.array([0, 0, 0])

        self.ax.quiver(*origin, *info["comming_refline_points"][0], color='r', label="X-axis")
        self.fig.canvas.draw()
        plt.pause(0.001)

class PrintVector2DToNextReferencePoint(TestEnvironmentCallback):
    def __init__(self):
        super().__init__()

        self.fig, self.ax = plt.subplots(figsize=(5,5))
        self._setup_plot()

        plt.ion()
        plt.show()

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        self.ax.cla()  # Clear previous plot
        self._setup_plot()  # Reset labels/aspect

        points :np.ndarray= info["comming_refline_points"]

        self.ax.scatter(points[0, 0], points[0, 2], color='green', marker='x', s=50)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _call_after_run(self):
        plt.ioff()
        plt.show()

    def _setup_plot(self):
        self.ax.set_title("XZ-View (Car Coordinate System)")
        self.ax.set_xlabel("X Axis")
        self.ax.set_ylabel("Z Axis")
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)


class Test_RefLine_Next_Point_Manager(TestEnvironmentCallback):
    def __init__(self, reference_line : np.ndarray):
        super().__init__()
        self.reference_line = reference_line
        # Set up a single 2D plot for the map
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self._setup_plot()

        plt.ion()
        plt.show()

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        points = info["comming_refline_points"]
        inv_orientation = info["orientation"].T
        position = info["position"]

        next_refline_index = info["rewards"]["nextpoint_reference_index"]

        # 1. Check determinant ≈ 1 since in_rotation should be a rotation matrix 
        det = np.linalg.det(inv_orientation)
        assert np.isclose(det, 1.0, atol=1e-5), f"Determinant not close to 1: det = {det}"
        # 2. Check if inv_orientation @ orientation ≈ Identity
        ident = inv_orientation @ info["orientation"]
        assert np.allclose(ident, np.eye(3), atol=1e-5), f"Matrix product not identity:\n{ident}"

        # Transform points to world coordinates
        points = (inv_orientation @ points.T).T + position

        # Clear and redraw plot
        self.ax.cla()
        self._setup_plot()

        # Plot reference line (black), transformed points (red), and position (green), and next reference-line-point (blue)
        self.ax.plot(-1. * self.reference_line[:, 0], self.reference_line[:, 2], color='black', linestyle='-')
        self.ax.plot(-1. * points[:, 0], points[:, 2], color='red', marker='o', linestyle='-')
        self.ax.scatter(-1. * position[0], position[2], color='green', marker='x', s=50)
        self.ax.scatter(-1. * self.reference_line[next_refline_index, 0], self.reference_line[next_refline_index, 2], color='blue', marker='x', s=50)
        
        self.fig.canvas.draw()
        plt.pause(0.001)

    def _setup_plot(self):
        self.ax.set_title("XZ View (World Space)")
        self.ax.set_xlabel("X Axis")
        self.ax.set_ylabel("Z Axis")
        self.ax.set_aspect('equal')

    """
    y_lim:
     - d = (0,2000)
     - drel = (0,1)
     - velocity_delta = (0,100)
    
    """

class Test_1D_Next_Point_Manager(TestEnvironmentCallback):
    def __init__(self, keys_to_plot, y_lim=(-10, 10)):
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

        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self._setup_plot()

        plt.ion()
        plt.show()

    def _setup_plot(self):
        self.ax.set_title("Debug Plot")
        self.ax.set_xlabel("Timestep")
        self.ax.set_ylabel("Value")
       # self.ax.set_ylim(*self.y_lim)

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        info["refline_idx"] = info["rewards"]["nextpoint_reference_index"]
        
        # Check and store values for all keys
        for key in self.keys_to_plot:
            assert key in info, f"The key '{key}' is not in the info dict"
            val = info[key]
            assert np.ndim(val) <= 1, f"The value for '{key}' must be 1D, got shape {np.shape(val)}"
            self.vals[key].append(val)

        # Clear and redraw the plot
        self.ax.cla()
        self._setup_plot()

        for key in self.keys_to_plot:
            data = np.array(self.vals[key])
            if data.ndim == 1:
                self.ax.plot(range(len(data)), data, label=key)
            else:
                for i in range(data.shape[1]):
                    self.ax.plot(range(len(data)), data[:, i], label=f"{key}[{i}]")

        self.ax.legend()
        plt.draw()
        plt.pause(0.001)


    def _setup_plot(self):
        self.ax.set_title(f"Tracking '{self.keys_to_plot}' Over Time")
        self.ax.set_xlabel("Step")
        #self.ax.set_ylabel(self.keys_to_plot)
        #self.ax.set_ylim(*self.y_lim)  # fixed Y-axis limits

class Test_3D_Next_Point_Manager(TestEnvironmentCallback):
    def __init__(self, key_to_plot, y_lim=(-1, 1)):
        super().__init__()
        self.key_to_plot = key_to_plot
        self.y_lim = y_lim
        self.vals = []

        # Set up 3 vertically stacked subplots for x, y, z
        self.fig, self.axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        self._setup_plot()

        plt.ion()
        plt.tight_layout()
        plt.show()

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        assert self.key_to_plot in info, f"The key '{self.key_to_plot}' is not in the info dict"

        val = np.asarray(info[self.key_to_plot])
        assert val.ndim == 1 and val.shape[0] == 3, \
            f"The value for '{self.key_to_plot}' must be a 3D vector, got shape {val.shape}"

        self.vals.append(val)

        # Clear subplots
        for ax in self.axes:
            ax.cla()

        # Convert to array for plotting
        vals_array = np.array(self.vals)  # shape: (steps, 3)
        labels = ['x', 'y', 'z']
        colors = ['red', 'green', 'blue']

        for i in range(3):
            self.axes[i].plot(vals_array[:, i], color=colors[i])
            self.axes[i].set_ylabel(labels[i])
            self.axes[i].set_ylim(*self.y_lim)
            self.axes[i].grid(True)

        self.axes[2].set_xlabel("Step")
        self.fig.suptitle(f"3D Vector Components of '{self.key_to_plot}'")

        plt.draw()
        plt.pause(0.001)

    def _setup_plot(self):
        labels = ['x', 'y', 'z']
        for i, ax in enumerate(self.axes):
            ax.set_ylabel(labels[i])
            ax.set_ylim(*self.y_lim)
            ax.grid(True)
        self.axes[2].set_xlabel("Step")

class Test_Lateral_Dist_Next_Point_Manager(TestEnvironmentCallback):
    def __init__(self, reference_line_manager):
        super().__init__()
        self.ref_line_manager =  reference_line_manager
        self.reference_line = self.ref_line_manager.reference_line

        # Plot 1: Map view (XZ)
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self._setup_plot()

        # Plot 2: Distance bar plot
        self.bar_fig, self.bar_ax = plt.subplots(figsize=(4, 4))
        self.bar_container = self.bar_ax.bar(["Distance"], [0.0], color='magenta')
        self.bar_ax.set_ylim(0, 10)  # Adjust max Y limit as needed
        self.bar_ax.set_ylabel("Euclidean Distance")
        
        #plot 3
        self.gauss_plot = plt.subplots(figsize=(6, 6))
        self.gauss_dist = norm(0,np.sqrt(12))
        # Create values for the smooth Gaussian curve
        self.x_vals = np.linspace(-50, 50, 500)
        self.y_vals = self.gauss_dist.pdf(self.x_vals)
        self.gauss_curve = self.gauss_dist.rvs(size=1000) 
        plt.ion()
        plt.show()

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        position = info["position"]
        next_refline_index = info["rewards"]["nextpoint_reference_index"]

        ref_line_point = self.reference_line[next_refline_index]
        diff_vec = ref_line_point - position
        lateral_distance = self.ref_line_manager.calculate_lateral_difference(next_refline_index,position)


        # Clear and redraw main map plot
        self.ax.cla()
        self._setup_plot()

        # Plot reference line, current position, and closest point
        self.ax.plot(-1. * self.reference_line[:, 0], self.reference_line[:, 2], color='black', linestyle='-')
        self.ax.scatter(-1. * position[0], position[2], color='green', marker='x', s=50)
        self.ax.scatter(-1. * ref_line_point[0], ref_line_point[2], color='blue', marker='x', s=50)

    
        # Compute difference vector and draw it as an arrow
        self.ax.arrow(-1. * position[0], position[2], -1. * diff_vec[0], diff_vec[2],
                      head_width=0.1, head_length=0.2, fc='magenta', ec='magenta', length_includes_head=True)

        # Draw updated main map
        self.fig.canvas.draw()

        # === Distance Bar Plot Update ===
        self.bar_container[0].set_height(lateral_distance)
        self.bar_ax.set_ylim(0, max(10, lateral_distance + 1))  # Auto-expand if needed
        self.bar_ax.set_title(f"Distance to Ref Point: {lateral_distance:.2f}")

        self.bar_fig.canvas.draw()

        prob = self.gauss_dist.pdf(lateral_distance)
        fig, ax = self.gauss_plot  # unpack the figure and axes
        ax.clear()  # clear previous plots if needed

        # Plot the Gaussian curve
        ax.plot(self.x_vals, self.y_vals, label='Gaussian PDF', color='blue')
        ax.axvline(lateral_distance, color='red', linestyle='--', label=f'Sample x = {lateral_distance}')
        ax.plot(lateral_distance, prob, 'ro')  # red dot at the point

        # Pause for real-time updates
        plt.pause(0.001)

    def _setup_plot(self):
        self.ax.set_title("XZ View (World Space)")
        self.ax.set_xlabel("X Axis")
        self.ax.set_ylabel("Z Axis")
        self.ax.set_aspect('equal')

class Test_Reward_Next_Point_Manager(TestEnvironmentCallback):
    def __init__(self, key_to_plot = None, y_lim=(-1, 1)):
        """
        All available reward keys :
            - accum_dist_reward  
            - race_not_finished_reward 
            - race_finished  
            - other_term_reward  
            - backward_punishment  
            - distance_to_center_reward  
            - velocity_change_reward
        If none is explicitly given, plots all rewards.
        """
        super().__init__()
        self.key_to_plot = key_to_plot
        self.y_lim = y_lim  # fixed y-axis scale
        self.vals = {}

        self.fig, self.ax = plt.subplots(figsize=(15, 15))
        self._setup_plot()

        plt.ion()
        plt.show()

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        rewards = info["rewards"]
        for k in rewards:
            if k in ["nextpoint_reference_index"]:
                continue
            if k not in self.vals: self.vals[k] = []
            self.vals[k].append(rewards[k])

        # Plot
        self.ax.cla()
        self._setup_plot()

        # Plot each reward key
        if self.key_to_plot == None:
            for key, values in self.vals.items():
                if key in ["nextpoint_reference_index"]: continue
                self.ax.plot(values, label=key)
        else:
            self.ax.plot(self.vals[self.key_to_plot], label=self.key_to_plot)

        self.ax.legend()
        plt.draw()
        plt.pause(0.001)

    def _setup_plot(self):
        if not self.key_to_plot == None:
            self.ax.set_title(f"Tracking '{self.key_to_plot}' Over Time")
            self.ax.set_ylabel(self.key_to_plot)
        else:
            self.ax.set_title(f"Tracking rewards over time")
            self.ax.set_ylabel("Reward Values per env step")
        self.ax.set_xlabel("Step")
        #self.ax.set_ylim(*self.y_lim)  # fixed Y-axis limits


class PretrainingDataCollection(TestEnvironmentCallback):
    def __init__(self, reference_line_manager : ReferenceLineManager, logging_directory : str, continuation_idx : int = -1):
        """
        Parameters
        -----------
            - reference_line_manager    : ReferenceLineManager used by environment
            - logging_directory         : Logging directory in which dataset is created or recorded
            - continueation_idx         : If this is set to some index other than -1, it is assumed there is already an existing dataset and
                                            the existing dataset is upposed to be extended. If it's -1, a new dataset is created (be careful; if it's -1 and there IS a dataset there, it'll be overwritten.)"""
        super().__init__()
        self.ref_line_manager = reference_line_manager
        self.logging_directory = logging_directory
        self.img_directory = os.path.join(self.logging_directory, "images")
        os.makedirs(self.img_directory, exist_ok = True)
        self.labels = os.path.join(self.logging_directory, "labels.csv")
        
        if not os.path.exists(self.labels) and continuation_idx == -1:
            with open(self.labels, "w") as file:
                file.write("filename,lateral_distance\n")

        if not continuation_idx == -1:
            self.n_step = continuation_idx
        else:
            assert os.path.exists(self.labels), "Expected dataset to already exists but did not find labels."


    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):

        i, d, drel = self.ref_line_manager.get_distance_to_next_point()
        if i == 0:
            self.n_step += 1
            return
        
        idx, lateral_dist = self.ref_line_manager.get_last_calculated_lateral_distance()
        assert i == idx, f"Expected indexes to be the same but got current refline idx {i} and index to which lateral distance was calculated at {idx}"
        with open(self.labels, "a") as file:
            file.write(f"img_{self.n_step}.npy,{lateral_dist}\n")

        img : np.ndarray = processed_obs["image"]
        np.save(os.path.join(self.img_directory, f"img_{self.n_step}.npy"), img)

        
        self.n_step += 1


class Plot_Obs_Images(TestEnvironmentCallback):
    def __init__(self):
        super().__init__()

        # Setup a single figure and axis for the image
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.image_handle = None  # Will hold the imshow image object

        plt.ion()
        plt.show()

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        # Get the image from observation (shape: (1, 1, H, W))
        img_tensor = processed_obs["image"]  # assumed torch.Tensor
        img_np = img_tensor.squeeze().cpu().numpy()  # shape: (H, W)

        if self.image_handle is None:
            # First time: create the imshow object
            self.image_handle = self.ax.imshow(img_np, cmap='gray')
            self.ax.axis('off')
        else:
            # Update image data
            self.image_handle.set_data(img_np)

        self.fig.canvas.draw()
        plt.pause(0.001)  # Small pause to allow GUI update



import multiprocessing as mp
from queue import Empty

class PlotterProcess(mp.Process):
    def __init__(self, data_queue, plotter):
        """
        Parameters:
            data_queue (mp.Queue): Queue receiving data to plot.
            plotter (EnvPlotter): An instance of a concrete EnvPlotter subclass.
        """
        super().__init__()
        self.queue = data_queue
        self.plotter = plotter

    def run(self):
        """
        Run the plotting loop in a separate process.
        """
        plt.ion()
        self.plotter.setup_plot()

        while True:
            try:
                data = self.queue.get(timeout=1)

                if data is None:
                    print("[PlotterProcess] Shutdown signal received.")
                    break  # Graceful shutdown

                # Drain any backlog, keeping the most recent item
                while not self.queue.empty(): data = self.queue.get_nowait() # NOTE this introduces skips, thing of removing this to prevent confusion
                self.plotter.plot(data)

            except Empty: continue

from trackmania_env.utils.environment_plots import Plot_Obs_Images,Plot_Rewards,Plot_Lateral_Distance,Plot_RefLine,PrintRotation,Plot_1D_Values

class NonBlockingPlot(TestEnvironmentCallback):
    def __init__(self, plotter):
        super().__init__()
        self.queue = mp.Queue()
        self.plot_process = PlotterProcess(data_queue=self.queue, plotter=plotter)
        self.plot_process.start()

    def __del__(self):
        try:
            self.queue.put(None)  # Signal to shutdown
            self.plot_process.join(timeout=1)
        except Exception:
            pass

class Plot_Obs_Images_Callback(NonBlockingPlot):
    def __init__(self):
        super().__init__(plotter=Plot_Obs_Images())

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        img_tensor = processed_obs["image"]
        self.queue.put(img_tensor)


BANNED =  ["nextpoint_reference_index"]
class Plot_Rewards_Callback(NonBlockingPlot):
    def __init__(self, key_to_plot=None, y_lim=(-1, 1)):
        super().__init__(plotter=Plot_Rewards(key_to_plot=key_to_plot, y_lim=y_lim))

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        rewards = info["rewards"]
        rewards = {k: v for k, v in rewards.items() if k not in BANNED}
        self.queue.put(rewards)

class Plot_Lateral_Distance_Callback(NonBlockingPlot):
    def __init__(self,reference_line_manager):
        self.data = {}
        super().__init__(plotter= Plot_Lateral_Distance(reference_line_manager))
    
    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        self.data["position"] = info["position"]
        self.data["next_refline_index"] = info["rewards"]["nextpoint_reference_index"]
        self.queue.put(self.data)

class Plot_ReferenceLine_Callback(NonBlockingPlot):
    def __init__(self,reference_line):
        self.data = {}
        super().__init__(plotter=Plot_RefLine(reference_line))
    
    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        self.data["position"] = info["position"]
        self.data["orientation"] = info["orientation"]
        self.data["comming_refline_points"] = info["comming_refline_points"]
        self.data["next_refline_index"] = info["rewards"]["nextpoint_reference_index"]
        self.queue.put(self.data)

class Plot_Rotation_Callback(NonBlockingPlot):
    def __init__(self,):
        super().__init__(plotter=PrintRotation())
    
    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        rot_matrix = np.array(info["rotation_matrix"])
        self.queue.put(rot_matrix)

class Plot_1D_Values_Callback(NonBlockingPlot):
    def __init__(self, keys_to_plot, y_lim = None):
        self.keys_to_plot = keys_to_plot
        self.data = {}
        super().__init__(plotter=Plot_1D_Values(keys_to_plot=keys_to_plot, y_lim = y_lim))
    
    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        for key in self.keys_to_plot:
            assert key in info, f"The key '{key}' is not in the info dict"
            val = info[key]
            assert np.ndim(val) <= 1, f"The value for '{key}' must be 1D, got shape {np.shape(val)}"
            self.data[key] = val
        self.queue.put(self.data)