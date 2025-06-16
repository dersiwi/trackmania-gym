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
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback

# imports for communication between TMInterface and environment
from game_interaction.game_instance_manager2 import GameInstanceManager
from game_interaction.run_multiprocess_wrapper import start_process_and_wait_for_startsignal
from game_interaction.process_wrapper import TMIProcessWrapper

# gymnasium environment wrapper 
from gymnasium import ObservationWrapper
from stable_baselines3.common.monitor import Monitor



from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
from configs.config import TrainConfig
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers.vector import FilterObservation
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "../configs",
    "config_name": "train.yaml",
}

from utils.hydra_wandb_utils import get_models


@hydra.main(**_HYDRA_PARAMS)
def main(cfg : TrainConfig):
    
    # Start Weights and Biases login
    run_id = ""
    if cfg.wandb.use :
        wandb.login()
        wandb.config = omegaconf.OmegaConf.to_container(cfg, resolve=True,throw_on_missing=True)
        run = wandb.init(
            entity=cfg.wandb.entity, 
            project=cfg.wandb.project,
            sync_tensorboard=True, 
            monitor_gym=True,  
            save_code=True)
        run_id = run.id

    num_envs = 2
    base_port = 8775
    ports = [base_port+i for i in range(num_envs)]
    # Instanciate num_envs GMI, TMNF-Environment and start TMi-Interaction processes via threadpool.
    with ThreadPoolExecutor(max_workers=len(ports)) as executor:
        func = partial(start_process_and_wait_for_startsignal,
                   platform=cfg.platforms,
                   gmi_cfg=cfg.gmi,
                   image_width=cfg.image.width,
                   image_height=cfg.image.height)
        queues_procs = list(executor.map(func, ports))
        
    try:
        tm_env = SubprocVecEnv(
            [make_env(cfg,control_queue,response_queue) for _, control_queue, response_queue  in queues_procs],
            start_method="fork",)
        # get algorithm and start learning process
        vision_model, model = get_models(cfg, tm_env, print_params = True,run_id=run_id)
        # for sb3 the type would be BaseCallBack. For other callbacks we would need to manually write the other types.
        # TODO check if callbacks have the same class they inherit from 
        callback = hydra.utils.instantiate(cfg.wandb_callbacks)(model_save_path=f"models/{run_id}")  if cfg.wandb.use else None  
        model.learn(**cfg.learn_args, callback=callback)
        model.save(os.path.join(HydraConfig.get().run.dir, "model"))

    except Exception as e:
        traceback.print_exc()

    except KeyboardInterrupt as kinterrupt:
        print("KeyboardInterrupt")

    finally:
        # Finalize training and close game all processes.
        for tmi_process, control_queue, response_queue in queues_procs:
            control_queue.put(TMIProcessWrapper.IPCCommands.get_end_syncloop_command(1000)) #1000 doesnt matter.
            tmi_process.join()

def make_env(cfg,control_queue, response_queue):
    def _init():
        tm_env = TMNF_Single_Agent_Env(
            command_queue=control_queue,
            response_queue=response_queue)
        for _, wrapper_conf in cfg.rl_env.wrappers.items():
            wrapper : ObservationWrapper = hydra.utils.instantiate(wrapper_conf)
            print(f"Wrapping environment in {wrapper.__class__.__name__}")
            tm_env = wrapper(env=tm_env)
        return tm_env
    return _init


    """ This the gymnasium way of defining vectorised envs but this is not compatible with sb3

        tm_env = SyncVectorEnv([lambda: partial(TMNF_Single_Agent_Env, control_queue, response_queue)() 
                              for _, control_queue, response_queue  in queues_procs])
        
        #TODO exchange this later through a config file
        tm_env = FilterObservation(env=tm_env,filter_keys= cfg.rl_env.wrappers.observations_filter.filter_keys)
        tm_env = Vec_PytorchWrapper(env=tm_env)
        tm_env = Vec_BGRA_to_RGB(env=tm_env)
        tm_env = Vec_TransformGrayscale(env=tm_env,keep_dim=True)
        """

        # apply (Observation)-wrappers to the environment
        #for _, wrapper_conf in cfg.rl_env.wrappers.items():
        #    wrapper : ObservationWrapper = hydra.utils.instantiate(wrapper_conf)
        #    print(f"Wrapping environment in {wrapper.__class__.__name__}")
        #    tm_env = wrapper(env=tm_env)


if __name__ == "__main__": 
    main()
