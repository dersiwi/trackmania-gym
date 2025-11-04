import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from trackmania_env.callbacks.core import TestEnvironmentCallback, Live3dPlotEnvironmentCallback
from trackmania_env.plotting.core import NonBlockingPlot
from trackmania_env.plotting.factory import PlottingFactory
from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env

class Plot_Obs_Images_Callback(NonBlockingPlot):
    def __init__(self, img_size:tuple[int,int], color_space:str, backend:str = "matplotlib"):
        super().__init__(plotter=PlottingFactory(factory_name= "image",backend = backend).create(img_size = img_size, color_space = color_space))

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        img = processed_obs["image"]
        assert isinstance(img, np.ndarray), "image must be a NumPy ndarray"
        self.queue.put(img)

BANNED =  ["nextpoint_reference_index"]
class Plot_Rewards_Callback(NonBlockingPlot):
    def __init__(self,env:TMNF_Single_Agent_Env, keys_to_plot:list[str]=None, y_lim=(-1, 1), plot_total : bool = True,backend:str = "matplotlib"):

        reward_terms: list[str] = [r.name for r in  env.rew_calculator.reward_terms]
        reward_terms.append("total") #TODO total is only created during runtime. for now it works but maybe come up with something better 
        if keys_to_plot is not None:
            assert set(keys_to_plot).issubset(reward_terms), f"{keys_to_plot} not found in reward terms"
            reward_terms = keys_to_plot
        super().__init__(plotter=PlottingFactory(factory_name="lines",backend= backend).create(keys_to_plot= reward_terms, title = "Rewards",ylabel= "Rewards"))

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        rewards = info["rewards"]
        rewards = {k: v for k, v in rewards.items() if k not in BANNED}
        self.queue.put(rewards)

class Plot_Lateral_Distance_Callback(NonBlockingPlot):
    def __init__(self,reference_line_manager,backend:str = "matplotlib"):
        self.data = {}
        super().__init__(plotter= PlottingFactory(factory_name="lateral_distance",backend = backend).create(reference_line_manager=reference_line_manager))

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        self.data["position"] = info["position"]
        self.data["next_refline_index"] = info["next_refline_index"] 
        self.queue.put(self.data)

class Plot_ReferenceLine_Callback(NonBlockingPlot):
    def __init__(self,reference_line):
        self.data = {}
        super().__init__(plotter=PlottingFactory(factory_name="ref_line").create(reference_line= reference_line))

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        self.data["position"] = info["position"]
        self.data["orientation"] = info["orientation"]
        self.data["comming_refline_points"] = info["comming_refline_points"]
        self.data["next_refline_index"] = info["next_refline_index"]
        self.queue.put(self.data)

class Plot_Rotation_Callback(NonBlockingPlot):
    def __init__(self,):
        super().__init__(plotter=PlottingFactory(factory_name="rotation").create())

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        rot_matrix = np.array(info["rotation_matrix"])
        self.queue.put(rot_matrix)

class Plot_1D_Values_Callback(NonBlockingPlot):
    def __init__(self, keys_to_plot, y_lim = None):
        self.keys_to_plot = keys_to_plot
        self.data = {}
        super().__init__(plotter=PlottingFactory(factory_name="lines").create(keys_to_plot=keys_to_plot, ylim = y_lim, title = "Bunch of 1D Values",ylabel = ""))

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        for key in self.keys_to_plot:
            assert key in info, f"The key '{key}' is not in the info dict"
            val = info[key]
            assert np.ndim(val) <= 1, f"The value for '{key}' must be 1D, got shape {np.shape(val)}"
            self.data[key] = val
        self.queue.put(self.data)

class Plot_3D_Values_Callback(NonBlockingPlot):
    def __init__(self, key_to_plot, y_lim= None):
        self.key_to_plot = key_to_plot
        self.data = {}
        super().__init__(plotter=Plot_3D_Values(key_to_plot=key_to_plot, y_lim=y_lim))

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        assert self.key_to_plot in info, f"The key '{self.key_to_plot}' is not in the info dict"
        val = np.asarray(info[self.key_to_plot])
        assert val.ndim == 1 and val.shape[0] == 3, \
                f"The value for '{self.key_to_plot}' must be a 3D vector, got shape {val.shape}"

        self.data[self.key_to_plot] = val
        self.queue.put(self.data)

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
