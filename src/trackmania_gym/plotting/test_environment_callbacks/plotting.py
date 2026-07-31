import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from trackmania_env.envs.info import EnvironmentInfo
from matplotlib import pyplot as plt
from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
from plotting.test_environment_callbacks.core import TestEnvironmentCallback, Live3dPlotEnvironmentCallback
from plotting.core import NonBlockingPlot





class Plot_Obs_Images_Callback(NonBlockingPlot):
    def __init__(self, img_size:tuple[int,int], color_space:str, backend:str = "matplotlib"):
        super().__init__(factory_name="image", backend=backend, create_args = {"img_size" : img_size, "color_space" : color_space})
    

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        assert "image" in processed_obs, (
            "No image observation found in the current environment. "
            "This may be intentional if the environment does not provide visual observations."
        )
        img = processed_obs["image"]
        assert isinstance(img, np.ndarray), "image must be a NumPy ndarray"
        self.queue.put(img)

BANNED =  ["nextpoint_reference_index"]
class Plot_Rewards_Callback(NonBlockingPlot):
    def __init__(self,env:TMNF_Single_Agent_Env, keys_to_plot:list[str]=None, y_lim=(-1, 1), plot_total : bool = True,backend:str = "matplotlib"):

        reward_terms: list[str] = [r.name for r in  env.rew_calculator.terms]
        reward_terms.append("total") #TODO total is only created during runtime. for now it works but maybe come up with something better 
        if keys_to_plot is not None:
            assert set(keys_to_plot).issubset(reward_terms), f"{keys_to_plot} not found in reward terms"
            reward_terms = keys_to_plot

        super().__init__(factory_name="lines", backend=backend, create_args=dict(keys_to_plot= reward_terms, title = "Rewards",ylabel= "Rewards"))

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        rewards = info[EnvironmentInfo.REWARDS]
        rewards = {k: v for k, v in rewards.items() if k not in BANNED}
        self.queue.put(rewards)

class Plot_Lateral_Distance_Callback(NonBlockingPlot):
    def __init__(self,reference_line_manager,backend:str = "matplotlib"):
        self.data = {}
        # TODO : Actually check if passing the reference line manager works. If only for the utility methods fine (but then why pass it in first place?), but not for data from environment
        super().__init__(factory_name="lateral_distance3", backend = backend, create_args=dict(reference_line_manager=reference_line_manager))

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        self.data[EnvironmentInfo.POSITION] = info[EnvironmentInfo.POSITION]
        self.data[EnvironmentInfo.NEXT_REFLINE_IDX] = info[EnvironmentInfo.NEXT_REFLINE_IDX] 
        self.queue.put(self.data)

class Plot_ReferenceLine_Callback(NonBlockingPlot):
    def __init__(self,reference_line):
        self.data = {}
        super().__init__(factory_name="lateral_distance3", backend = "matplotlib", create_args=dict(reference_line= reference_line))

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        self.data[EnvironmentInfo.POSITION] = info[EnvironmentInfo.POSITION]
        self.data[EnvironmentInfo.ORIENTATION] = info[EnvironmentInfo.ORIENTATION]
        self.data[EnvironmentInfo.COMING_REFLINE_POINTS] = info[EnvironmentInfo.COMING_REFLINE_POINTS]
        self.data[EnvironmentInfo.NEXT_REFLINE_IDX] = info[EnvironmentInfo.NEXT_REFLINE_IDX]
        self.queue.put(self.data)

class Plot_Rotation_Callback(NonBlockingPlot):
    def __init__(self,):
        super().__init__(factory_name="rotation", backend = "matplotlib", create_args=dict())

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        rot_matrix = np.array(info["rotation_matrix"])
        self.queue.put(rot_matrix)

class Plot_1D_Values_Callback(NonBlockingPlot):
    """Plots the list of 1D values specified by keys_to_plot if they are stored in the info dict of the environment"""
    def __init__(self, keys_to_plot, y_lim = None):
        self.keys_to_plot = keys_to_plot
        self.data = {}
        super().__init__(factory_name="lines", backend = "matplotlib", create_args=dict(keys_to_plot=keys_to_plot, ylim = y_lim, title = "Bunch of 1D Values", ylabel = ""))

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        for key in self.keys_to_plot:
            assert key in info, f"The key '{key}' is not in the info dict"
            val = info[key]
            assert np.ndim(val) <= 1, f"The value for '{key}' must be 1D, got shape {np.shape(val)}"
            self.data[key] = val
        self.queue.put(self.data)

class Plot_3D_Value_Callback(NonBlockingPlot):
    """Plots the 3D value specified by key_to_plot. Important this plots only one key"""
    def __init__(self, key_to_plot:str, y_lim= None):
        self.key_to_plot = key_to_plot
        self.keys_to_plot = [ax+"-"+ self.key_to_plot for ax in ["x","y","z"]]
        self.data = {}

        super().__init__(factory_name="lines", backend = "matplotlib", 
                         create_args=dict(keys_to_plot=self.keys_to_plot,ylim=y_lim,title = f"Plot of the xyz components of {self.key_to_plot}",ylabel = ""))

    def _call_after_step(self, processed_obs, reward, terminated, truncated, info):
        assert self.key_to_plot in info, f"The key '{self.key_to_plot}' is not in the info dict"
        val = np.asarray(info[self.key_to_plot])
        
        assert val.ndim == 1 and val.shape[0] == 3, \
                f"The value for '{self.key_to_plot}' must be a 3D vector, got shape {val.shape}"
        for i in range(len(self.keys_to_plot)) : self.data[self.keys_to_plot[i]] = val[i]
        self.queue.put(self.data)
