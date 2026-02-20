import sys, os

sys.path.append(os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__), '..'), '..'))) # TODO : <- i don't want this here and it shouldnt have to be here!!!

from game_interaction.ipc_fields import IPCCommands

from game_interaction.process_management import ProcessManagement
from trackmania_env.envs.enivonrments import get_single_environment

from trackmania_env.envs.testenv_single_agent import TestEnvironment

import plotting.test_environment_callbacks.plotting as plot_callback
import plotting.test_environment_callbacks.printing as print_callback

from utils.hydra_wandb_utils import load_and_merge_platform

import hydra

from configs.config import TrainConfig

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "../../configs",
    "config_name": "train.yaml",
}

@hydra.main(**_HYDRA_PARAMS)
def main(cfg : TrainConfig):
    cfg = load_and_merge_platform(cfg)

    pm = ProcessManagement(cfg, cfg.rl_env.obs_manager.img_width, cfg.rl_env.obs_manager.img_height)
    ipcommandsender = pm.start_process_and_wait_for_startsignal()
    tm_env : TestEnvironment = get_single_environment(cfg, ipcommandsender, test=True)

    obs, info = tm_env.reset()

    # Plotting referece line 
    # tm_env.add_env_test_calback(plot_callback.Plot_ReferenceLine_Callback(reference_line= tm_env.reference_line.reference_line))

    # Plotting lateral distance 
    #tm_env.add_env_test_calback(plot_callback.Plot_Lateral_Distance_Callback(reference_line_manager=tm_env.env.reference_line))
    
    # Plotting Images
    obs_manager = cfg.rl_env.obs_manager 
    #tm_env.add_env_test_calback(plot_callback.Plot_Obs_Images_Callback(img_size= (obs_manager.img_width, obs_manager.img_height), color_space= obs_manager.colorspace, backend ="matplotlib"))
    
    # Plotting Rewards 
    tm_env.add_env_test_calback(plot_callback.Plot_Rewards_Callback(env = tm_env))

    # Plotting arbitrary 1D values
    # keys_to_plot = ["last_has_any_lateral_contact_time","gas","display_speed"]
    # tm_env.add_env_test_calback(plot_callback.Plot_1D_Values_Callback(keys_to_plot= keys_to_plot))

    #Plotting a specified 3D value 
    # tm_env.add_env_test_calback(plot_callback.Plot_3D_Value_Callback(key_to_plot="velocity"))

    tm_env.step_with_manual_input()
    
    pm.finalize_processes()

if __name__ == "__main__":
    main()
