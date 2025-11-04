import sys, os

sys.path.append(os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__), '..'), '..'))) # TODO : <- i don't want this here and it shouldnt have to be here!!!

from game_interaction.ipc_fields import IPCCommands

from game_interaction.run_multiprocess_wrapper import start_process_and_wait_for_startsignal
from trackmania_env.envs.enivonrments import get_environment

from trackmania_env.envs.testenv_single_agent import TestEnvironment
import trackmania_env.envs.testcases_single_agent as testcases

import trackmania_env.callbacks.plotting as plot_callback
import trackmania_env.callbacks.printing as print_callback

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

    tmi_process, control_queue, response_queue = start_process_and_wait_for_startsignal(cfg,cfg.rl_env.obs_manager.img_width, cfg.rl_env.obs_manager.img_height)
    tm_env : TestEnvironment = get_environment(cfg, control_queue, response_queue, test=True)

    obs, info = tm_env.reset()

    # Plotting referece line 
    # tm_env.add_env_test_calback(plot_callback.Plot_ReferenceLine_Callback(reference_line= tm_env.reference_line.reference_line))

    # Plotting lateral distance 
    # tm_env.add_env_test_calback(plot_callback.Plot_Lateral_Distance_Callback(reference_line_manager=tm_env.reference_line))
    
    # Plotting Images
    # obs_manager = cfg.rl_env.obs_manager 
    # tm_env.add_env_test_calback(plot_callback.Plot_Obs_Images_Callback(img_size= (obs_manager.img_width,obs_manager.img_height), color_space= obs_manager.colorspace, backend ="matplotlib"))
    
    # Plotting Rewards 
    tm_env.add_env_test_calback(plot_callback.Plot_Rewards_Callback(env = tm_env))

    tm_env.step_with_manual_input()
    
    control_queue.put(IPCCommands.get_end_syncloop_command(1000)) #1000 doesnt matter.
    tmi_process.join()

if __name__ == "__main__":
    main()
