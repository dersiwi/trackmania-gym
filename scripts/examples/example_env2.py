import sys, os

sys.path.append(os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__), '..'), '..'))) # TODO : <- i don't want this here and it shouldnt have to be here!!!

from game_interaction.process_wrapper import TMIProcessWrapper

from game_interaction.run_multiprocess_wrapper import start_process_and_wait_for_startsignal
from trackmania_env.observations.observations import get_observation_manager
from trackmania_env.rewards.getrewards import get_reward_calculator

from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
from trackmania_env.envs.testenv_single_agent import TestEnvironment
import trackmania_env.envs.testcases_single_agent as testcases

from trackmania_env.utils.actionmap import get_reverse_action_map
from trackmania_env.utils.reference_line_manager import ReferenceLineManager

import hydra
from configs.config import TrainConfig

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "../../configs",
    "config_name": "train.yaml",
}

@hydra.main(**_HYDRA_PARAMS)
def main(cfg : TrainConfig):

    tmi_process, control_queue, response_queue = start_process_and_wait_for_startsignal(cfg.platforms, cfg.gmi, cfg.image.width, cfg.image.height)

    tm_env = TestEnvironment(
        command_queue=control_queue,
        response_queue=response_queue,
        obs_manager=get_observation_manager(cfg), 
        reward_calculator=get_reward_calculator(cfg),
        reference_line=  ReferenceLineManager(cfg.gmi.reference_line),
        env_cfg=cfg.rl_env.env,
        platform=cfg.platforms.os)
    
    tm_env.add_env_test_calback(testcases.PrintRewardsToConsole())
    #tm_env.add_env_test_calback(testcases.Test_RefLine_Next_Point_Manager(tm_env.reference_line.reference_line))
    #tm_env.add_env_test_calback(testcases.Test_RefLine_Next_Point_Manager())

    #tm_env.add_env_test_calback(testcases.Test_1D_Next_Point_Manager(key_to_plot="refline_idx",y_lim=(0,3540)))
    #tm_env.add_env_test_calback(testcases.PrintVector2DToNextReferencePoint())
    #tm_env.add_env_test_calback(testcases.PrintVectorToNextReferencePoint())
    #tm_env.add_env_test_calback(testcases.Test_Lateral_Dist_Next_Point_Manager(tm_env.reference_line.reference_line))
    #tm_env.add_env_test_calback(testcases.Test_Reward_Next_Point_Manager())
    #tm_env.add_env_test_calback(testcases.Test_3D_Next_Point_Manager(key_to_plot="velocity_delta",y_lim=(-50,50)))
    tm_env.step_with_manual_input()
    
    control_queue.put(TMIProcessWrapper.IPCCommands.get_end_syncloop_command(1000)) #1000 doesnt matter.
    tmi_process.join()

if __name__ == "__main__":
    main()