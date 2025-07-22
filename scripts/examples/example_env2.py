import sys, os

sys.path.append(os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__), '..'), '..'))) # TODO : <- i don't want this here and it shouldnt have to be here!!!

from game_interaction.ipc_fields import IPCCommands

from game_interaction.run_multiprocess_wrapper import start_process_and_wait_for_startsignal
from trackmania_env.envs.enivonrments import get_environment

from trackmania_env.envs.testenv_single_agent import TestEnvironment
import trackmania_env.envs.testcases_single_agent as testcases


import hydra
from configs.config import TrainConfig

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "../../configs",
    "config_name": "train.yaml",
}

@hydra.main(**_HYDRA_PARAMS)
def main(cfg : TrainConfig):

    tmi_process, control_queue, response_queue = start_process_and_wait_for_startsignal(cfg,cfg.rl_env.obs_manager.img_width, cfg.rl_env.obs_manager.img_height)
    tm_env : TestEnvironment = get_environment(cfg, control_queue, response_queue, test=True)

    obs, info = tm_env.reset()
    #tm_env.add_env_test_calback(testcases.PrintRewardsToConsole())
    #tm_env.add_env_test_calback(testcases.Test_RefLine_Next_Point_Manager(tm_env.reference_line.reference_line))

    #tm_env.add_env_test_calback(testcases.Test_1D_Next_Point_Manager(key_to_plot="refline_idx",y_lim=(0,3540)))
    #tm_env.add_env_test_calback(testcases.PrintVector2DToNextReferencePoint())
    #tm_env.add_env_test_calback(testcases.PrintVectorToNextReferencePoint())
    #tm_env.add_env_test_calback(testcases.Test_Lateral_Dist_Next_Point_Manager(tm_env.reference_line))
    #tm_env.add_env_test_calback(testcases.Test_Reward_Next_Point_Manager2())
    #tm_env.add_env_test_calback(testcases.Test_3D_Next_Point_Manager(key_to_plot="velocity_delta",y_lim=(-50,50)))
    #tm_env.add_env_test_calback(testcases.PretrainingDataCollection(reference_line_manager=tm_env.reference_line, logging_directory=r"C:\Users\siwis\Documents\dataset", continuation_idx=1721))
    #tm_env.add_env_test_calback(testcases.Plot_1D_Values_Callback(keys_to_plot=["display_speed","gas","last_has_any_lateral_contact_time"]))
    #tm_env.add_env_test_calback(testcases.Plot_Lateral_Distance_Callback(tm_env.reference_line))
    tm_env.add_env_test_calback(testcases.Plot_3D_Values_Callback(key_to_plot= "position"))



    tm_env.step_with_manual_input()
    
    control_queue.put(IPCCommands.get_end_syncloop_command(1000)) #1000 doesnt matter.
    tmi_process.join()

if __name__ == "__main__":
    main()