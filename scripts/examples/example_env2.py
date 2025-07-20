import sys, os

sys.path.append(os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__), '..'), '..'))) # TODO : <- i don't want this here and it shouldnt have to be here!!!

from game_interaction.ipc_fields import IPCCommands

from game_interaction.run_multiprocess_wrapper import start_process_and_wait_for_startsignal
from trackmania_env.observations.observations import get_observation_manager
from trackmania_env.rewards.getrewards import get_reward_calculator
from trackmania_env.terminations.get_termination_manager import get_termination_manager

from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
from trackmania_env.envs.testenv_single_agent import TestEnvironment
import trackmania_env.envs.testcases_single_agent as testcases

from trackmania_env.utils.actionmap import get_reverse_action_map
from trackmania_env.utils.reference_line_manager import ReferenceLineManager
from trackmania_env.utils.orientationless_random_respawn_manager import OrientationlessRespawnManager

import hydra
from configs.config import TrainConfig

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "../../configs",
    "config_name": "train.yaml",
}

@hydra.main(**_HYDRA_PARAMS)
def main(cfg : TrainConfig):

    tmi_process, control_queue, response_queue = start_process_and_wait_for_startsignal(cfg, cfg.image.width, cfg.image.height)
    obs_manager = get_observation_manager(cfg = cfg, wrap_obs_in_test = False)
    reward_calculator = get_reward_calculator(reward_calculator_cfg = cfg.rl_env.reward_manager)
    termination_manger = get_termination_manager(termination_cfg= cfg.rl_env.termination_manager)

    tm_env = TestEnvironment(
        command_queue=control_queue,
        response_queue=response_queue,
        obs_manager=obs_manager, 
        reward_calculator= reward_calculator,
        termination_manger= termination_manger,
        reference_line=ReferenceLineManager(cfg.gmi.reference_line),
        env_cfg=cfg.rl_env.env,
        platform=cfg.platforms.os)
    
    reward_calculator.set_env(tm_env)
    obs_manager.set_env(tm_env)
    termination_manger.set_env(tm_env)

    tm_env.orientationless_respawn_manager = OrientationlessRespawnManager(respawn_coordinates=OrientationlessRespawnManager.get_respawns_for_very_long_checkpoints())

    
    #tm_env.add_env_test_calback(testcases.PrintRewardsToConsole())
    #tm_env.add_env_test_calback(testcases.Test_RefLine_Next_Point_Manager(tm_env.reference_line.reference_line))

    #tm_env.add_env_test_calback(testcases.Test_1D_Next_Point_Manager(key_to_plot="refline_idx",y_lim=(0,3540)))
    #tm_env.add_env_test_calback(testcases.PrintVector2DToNextReferencePoint())
    #tm_env.add_env_test_calback(testcases.PrintVectorToNextReferencePoint())
    #tm_env.add_env_test_calback(testcases.Test_Lateral_Dist_Next_Point_Manager(tm_env.reference_line))
    #tm_env.add_env_test_calback(testcases.Test_Reward_Next_Point_Manager())
    #tm_env.add_env_test_calback(testcases.Test_3D_Next_Point_Manager(key_to_plot="velocity_delta",y_lim=(-50,50)))
    #tm_env.add_env_test_calback(testcases.PretrainingDataCollection(reference_line_manager=tm_env.reference_line, logging_directory=r"C:\Users\siwis\Documents\dataset", continuation_idx=1721))
    tm_env.add_env_test_calback(testcases.Plot_Obs_Images())

    tm_env.step_with_manual_input()
    
    control_queue.put(IPCCommands.get_end_syncloop_command(1000)) #1000 doesnt matter.
    tmi_process.join()

if __name__ == "__main__":
    main()