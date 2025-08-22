
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # TODO : <- i don't want this here and it shouldnt have to be here!!!

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from configs.config import TrainConfig
import traceback



from game_interaction.run_multiprocess_wrapper import start_process_and_wait_for_startsignal
from game_interaction.ipc_fields import IPCCommands

from trackmania_env.envs.enivonrments import get_environment

from utils.hydra_wandb_utils import get_models, load_and_merge_yaml

run = "Level1"
run_path_hydra = f"C:\\Users\\siwis\\OneDrive\\Dokumente\\Studium\\Master\\1.Semester\\trackmania\\interaction_template\\outputs\\{run}\\.hydra"
model_path = f"C:\\Users\\siwis\\OneDrive\\Dokumente\\Studium\\Master\\1.Semester\\trackmania\\interaction_template\\outputs\\{run}\\best_model.zip"
#cfg : TrainConfig= OmegaConf.load(os.path.join("configs", "train.yaml"))

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": run_path_hydra,
    "config_name": "config.yaml",
}

@hydra.main(**_HYDRA_PARAMS)
def main(cfg : TrainConfig):
    cfg = load_and_merge_yaml(cfg, cfg.platforms_config_path)
    tmi_process, control_queue, response_queue = start_process_and_wait_for_startsignal(cfg, cfg.rl_env.obs_manager.img_width, cfg.rl_env.obs_manager.img_height)

    try:
        tm_env = get_environment(cfg, control_queue, response_queue)
        
        # get algorithm and start learning process
        vision_model, model = get_models(cfg, tm_env, print_params = True,load_model_path= model_path)
        #model.policy.features_extractor.eval()
        # this should be sufficient 
        eval_policy = model.policy.eval()
        
        terminated = False
        observations, info = tm_env.reset()
        while True:
            action, state = eval_policy.predict(observations, deterministic=True)
            observations, reward, terminated, truncated, info = tm_env.step(action)
            if terminated or truncated: tm_env.reset()
    except Exception as e:
        traceback.print_exc()

    except KeyboardInterrupt as kinterrupt:
        print("KeyboardInterrupt")

    finally:
        # Finalize training and close game all processes.
        control_queue.put(IPCCommands.get_end_syncloop_command(1000)) 
        tmi_process.join()
        

if __name__ == "__main__":
    main()