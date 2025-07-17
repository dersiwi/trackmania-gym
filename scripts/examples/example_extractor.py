import sys, os
from pathlib import Path
import gymnasium as gym
from contextlib import redirect_stdout
import numpy as np
from multiprocessing import Queue, Process
import time 
import logging


sys.path.append(os.path.abspath(os.path.join(
    os.path.join(os.path.dirname(__file__), '..'), '..'))) # TODO : <- i don't want this here and it shouldnt have to be here!!!


from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
from trackmania_env.envs.testenv_single_agent import TestEnvironment, TestLinesightRewards, PrintRewardsToConsole

from trackmania_env.utils.actionmap import get_reverse_action_map



from game_interaction.ipc_fields import IPCCommands


from game_interaction.run_multiprocess_wrapper import start_process_and_wait_for_startsignal
from trackmania_env.observations.observations import get_observation_manager
from trackmania_env.rewards.getrewards import get_reward_calculator

from neuronal_networks.vision_encoder.prebuilt_nets import PrebuiltnetImplementation
from neuronal_networks.custom_extractor import TMN_Extractor

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
        env_cfg=cfg.rl_env.env,
        platform=cfg.platforms.os)
    
    tm_env.add_env_test_calback(TestLinesightRewards())
    tm_env.add_env_test_calback(PrintRewardsToConsole())
    
    tm_env.step_with_manual_input()
    
    control_queue.put(IPCCommands.get_end_syncloop_command(1000)) #1000 doesnt matter.
    tmi_process.join()

    out_dim = 10
    vision_model = PrebuiltnetImplementation("resnet18", "resnet", out_dims=out_dim,pretrained=True,
                                    in_color_channels=tm_env.observation_space["image"].shape[-1])
    extractor = TMN_Extractor(tm_env.observation_space,vision_model=vision_model,vision_model_out_dim=out_dim)
    with open('example_extractor_out.txt', 'w') as f:
        with redirect_stdout(f):
            for i in range(20):
                obs,_,_,_,_ = tm_env.step(np.random.randint(0,12))
                v = extractor(obs)
                print(v)
                print(v.shape)
                print("-"*20)


if __name__ == "__main__":
    main()





