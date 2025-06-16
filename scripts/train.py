# utils
import sys, os
# TODO : <- i don't want this here and it shouldnt have to be here!!!
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

# Hydra related imports
import hydra
import omegaconf
from hydra.core.hydra_config import HydraConfig
import traceback

# Weights and Biases related imports
import wandb

# imports for communication between TMInterface and environment
from game_interaction.run_multiprocess_wrapper import start_process_and_wait_for_startsignal
from game_interaction.process_wrapper import TMIProcessWrapper

# gymnasium environment wrapper 
from gymnasium import ObservationWrapper
from stable_baselines3.common.monitor import Monitor

# extractor imports
from stable_baselines3.common.base_class import BaseAlgorithm
from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
from trackmania_env.observations.observation_manager import ObservationManager
from trackmania_env.observations.observation_test import ObservationTest
 
from trackmania_env.observations.linesight_obs_wrapper import LinesightObservationWrapper

from configs.config import TrainConfig
from utils.printutils import print_model_params
from trackmania_env.utils.init_linesight_obs import get_linesight_obs_instance

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "../configs",
    "config_name": "train.yaml",
}

from utils.hydra_utils import get_models

@hydra.main(**_HYDRA_PARAMS)
def main(cfg : TrainConfig):
    
    # Start Weights and Biases login
    run_id = ""
    if cfg.wandb.use :
        wandb.login()
        wandb_conf = omegaconf.OmegaConf.to_container(cfg, resolve=True,throw_on_missing=True)
        wandb.config = wandb_conf
        run = wandb.init(
            entity=cfg.wandb.entity, 
            project=cfg.wandb.project,
            sync_tensorboard=True, 
            monitor_gym=True,  
            save_code=True,
            config=wandb_conf)
        run_id = run.id

    # Instanciate GMI, TMNF-Environment and start TMi-Interaction process.

    tmi_process, control_queue, response_queue = start_process_and_wait_for_startsignal(cfg.platforms, cfg.gmi, cfg.image.width, cfg.image.height)

    try:
        obs_manager_cfg = cfg.rl_env.obsmanager
        if cfg.rl_env.env.obs_manager =="basic":
            obs_manager = ObservationManager(observation_list=obs_manager_cfg.observation_list, 
                                            colorspace=obs_manager_cfg.colorspace,
                                            convert_torch=obs_manager_cfg.convert_torch,
                                            img_width=cfg.image.width, 
                                            img_height=cfg.image.height)
        elif cfg.rl_env.env.obs_manager =="linesight":
            obs_manager = get_linesight_obs_instance(cfg)
        elif cfg.rl_env.env.obs_manager == "test":
            obs_manager = ObservationTest(observation_list=obs_manager_cfg.observation_list, 
                                            colorspace=obs_manager_cfg.colorspace,
                                            convert_torch=obs_manager_cfg.convert_torch,
                                            img_width=cfg.image.width, 
                                            img_height=cfg.image.height,
                                            log_directory="logs/observations", log_frequency=30)
            
        if cfg.rl_env.env.test:
            from trackmania_env.envs.testenv_single_agent import TestEnvironment
            tm_env = TestEnvironment(command_queue=control_queue,
                                            response_queue=response_queue, 
                                            obs_manager=obs_manager,
                                            max_steps_before_reset=cfg.rl_env.env.max_steps_until_reset,
                                            game_speed=cfg.rl_env.env.game_speed)

        else:
            tm_env = TMNF_Single_Agent_Env(command_queue=control_queue,
                                            response_queue=response_queue, 
                                            obs_manager=obs_manager,
                                            max_steps_before_reset=cfg.rl_env.env.max_steps_until_reset,
                                            game_speed=cfg.rl_env.env.game_speed)
             
        # apply (Observation)-wrappers to the environment
        for _, wrapper_conf in cfg.rl_env.wrappers.items():
            wrapper : ObservationWrapper = hydra.utils.instantiate(wrapper_conf)
            print(f"Wrapping environment in {wrapper.__class__.__name__}")
            tm_env = wrapper(env=tm_env)
        
        # get algorithm and start learning process
        vision_model, model = get_models(cfg, tm_env, print_params = True,run_id=run_id)
        # for sb3 the type would be BaseCallBack. For other callbacks we would need to manually write the other types.
        # TODO check if callbacks have the same class they inherit from 
        callback = hydra.utils.instantiate(cfg.wandb_callbacks)(model_save_path=f"models/{run_id}")  if cfg.wandb.use else None  
        model.learn(**cfg.learn_args, callback=callback)
        if cfg.wandb.use:
            run.finish()
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