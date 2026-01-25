import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # TODO : <- i don't want this here and it shouldnt have to be here!!!

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from configs.config import TrainConfig
import traceback

from game_interaction.run_multiprocess_wrapper import start_process_and_wait_for_startsignal
from game_interaction.ipc_fields import IPCCommands


from utils.hydra_wandb_utils import get_models, load_and_merge_yaml
from utils.plotting.conv_NN_plot import VerboseExecution
from trackmania_env.envs.enivonrments import get_environment

run_path = "outputs/2025-06-03/14-34-38"
run_path_hydra = r"C:\Users\siwis\OneDrive\Dokumente\Studium\Master\1.Semester\trackmania\interaction_template\outputs\.hydra"
model_path = r"C:\Users\siwis\OneDrive\Dokumente\Studium\Master\1.Semester\trackmania\interaction_template\outputs\best_model.zip"
cfg : TrainConfig= OmegaConf.load(os.path.join("configs", "train.yaml"))

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": run_path_hydra,
    "config_name": "config.yaml",
}

@hydra.main(**_HYDRA_PARAMS)
def main(cfg : TrainConfig):
    cfg = load_and_merge_yaml(cfg, cfg.platforms_config_path)
    tmi_process, control_queue, response_queue = start_process_and_wait_for_startsignal(cfg, cfg.image.width, cfg.image.height)

    try:
        tm_env = get_environment(cfg, control_queue, response_queue)
        
        # get algorithm and start learning process
        vision_model, model = get_models(cfg, tm_env, print_params = True,load_model_path= model_path)
        verbose_model = VerboseExecution(model.policy.features_extractor.extractors["image"])
        model.policy.eval()
        terminated = False
        observations, info = tm_env.reset()
        while True:
            action, state = model.predict(observations, deterministic=True)
            observations, reward, terminated, truncated, info = tm_env.step(action)
            verbose_model.visualize(num_maps=6,num_rows=4)
            if terminated or truncated:
                tm_env.reset()

    except Exception as e:
        traceback.print_exc()

    except KeyboardInterrupt as kinterrupt:
        print("KeyboardInterrupt")

    finally:
        # Finalize training and close game all processes.
        control_queue.put(IPCCommands.get_end_syncloop_command()) 
        tmi_process.join()
        

if __name__ == "__main__":
    main()
