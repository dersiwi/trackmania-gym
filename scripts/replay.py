
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # TODO : <- i don't want this here and it shouldnt have to be here!!!

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from configs.config import TrainConfig
import traceback

from trackmania_env.envs.testenv_single_agent import TestEnvironment
from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
from game_interaction.game_instance_manager2 import GameInstanceManager
from game_interaction.run_multiprocess_wrapper import start_process_and_wait_for_startsignal
from game_interaction.process_wrapper import TMIProcessWrapper
from gymnasium import ObservationWrapper
from trackmania_env.rewards.getrewards import get_reward_calculator
from trackmania_env.observations.observations import get_observation_manager


from utils.hydra_wandb_utils import get_models


run_path = "outputs/2025-06-03/14-34-38"
run_path_hydra = os.path.join(run_path, ".hydra")
model_path = os.path.join("logs/downloaded/", "checkpoint_800000_steps.zip")

cfg : TrainConfig= OmegaConf.load(os.path.join("configs", "train.yaml"))
_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "../configs",
    "config_name": "train.yaml",
}
@hydra.main(**_HYDRA_PARAMS)
def main(cfg : TrainConfig):

    tmi_process, control_queue, response_queue = start_process_and_wait_for_startsignal(cfg.platforms, cfg.gmi, cfg.image.width, cfg.image.height)

    try:

        obs_manager = get_observation_manager(cfg)
            
        if cfg.rl_env.env.test:
            TM_ENV_CLASS = TestEnvironment
        else:
            TM_ENV_CLASS = TMNF_Single_Agent_Env

        tm_env = TM_ENV_CLASS(command_queue=control_queue,
                                response_queue=response_queue, 
                                obs_manager=obs_manager,
                                reward_calculator=get_reward_calculator(cfg),
                                env_cfg=cfg.rl_env.env)
        #gym.make_vec("TMNF_Single_Agent_ENV_v0",num_envs=2,)

        
        # apply (Observation)-wrappers to the environment
        for _, wrapper_conf in cfg.rl_env.wrappers.items():
            wrapper : ObservationWrapper = hydra.utils.instantiate(wrapper_conf)
            print(f"Wrapping environment in {wrapper.__class__.__name__}")
            tm_env = wrapper(env=tm_env)
        
        # get algorithm and start learning process
        vision_model, model = get_models(cfg, tm_env, print_params = True)


        model.load(model_path)
        
        terminated = False
        observations, info = tm_env.reset()
        while not terminated:
            action, state = model.predict(observations)
            observations, reward, terminated, truncated, info = tm_env.step(action)

    except Exception as e:
        traceback.print_exc()

    except KeyboardInterrupt as kinterrupt:
        print("KeyboardInterrupt")

    finally:
        # Finalize training and close game all processes.
        control_queue.put(TMIProcessWrapper.IPCCommands.get_end_syncloop_command(1000)) 
        tmi_process.join()
        

if __name__ == "__main__":
    main()