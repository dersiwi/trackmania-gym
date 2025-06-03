# utils
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # TODO : <- i don't want this here and it shouldnt have to be here!!!

import logging
import hydra
from hydra.core.hydra_config import HydraConfig
import traceback

# imports for communication between TMInterface and environment
from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
from game_interaction.game_instance_manager2 import GameInstanceManager
from game_interaction.run_multiprocess_wrapper import start_process_and_wait_for_startsignal
from game_interaction.process_wrapper import TMIProcessWrapper

# gymnasium environment wrapper 
from gymnasium import ObservationWrapper
from trackmania_env.wrappers.observations_filter import ObservationFilter
from trackmania_env.wrappers.bgra_to_rgb import BGRA_to_RGB
from trackmania_env.wrappers.transform_grayscale import TransformGrayscale
from trackmania_env.wrappers.transform_torch import PytorchWrapper
from trackmania_env.wrappers.observations_filter import ObservationFilter

from configs.config import TrainConfig

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "../configs",
    "config_name": "train.yaml",
}

from utils.hydra_utils import get_models

@hydra.main(**_HYDRA_PARAMS)
def main(cfg : TrainConfig):
    
    # Instanciate GMI, TMNF-Environment and start TMi-Interaction process.

    tmi_process, control_queue, response_queue = start_process_and_wait_for_startsignal(cfg.platforms, cfg.gmi, cfg.image.width, cfg.image.height)

    try:
        tm_env = TMNF_Single_Agent_Env(
            command_queue=control_queue,
            response_queue=response_queue)
        
        # apply (Observation)-wrappers to the environment
        for _, wrapper_conf in cfg.rl_env.wrappers.items():
            wrapper : ObservationWrapper = hydra.utils.instantiate(wrapper_conf)
            print(f"Wrapping environment in {wrapper.__class__.__name__}")
            tm_env = wrapper(env=tm_env)
        
        # get algorithm and start learning process
        vision_model, model = get_models(cfg, tm_env, print_params = True)
        model.learn(cfg.total_timesteps)
        model.save(os.path.join(HydraConfig.get().run.dir, "model"))

    except Exception as e:
        traceback.print_exc()

    except KeyboardInterrupt as kinterrupt:
        print("KeyboardInterrupt")

    finally:
        # Finalize training and close game all processes.
        control_queue.put(TMIProcessWrapper.IPCCommands.get_end_syncloop_command(1000)) #1000 doesnt matter.
        tmi_process.join()


if __name__ == "__main__": 
    main()
    