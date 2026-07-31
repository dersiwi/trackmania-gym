import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from configs.config import TrainConfig
import traceback

from trackmania_gym.utils.hydra_wandb_utils import  load_and_merge_platform
from trackmania_gym.tmn_sb3.utils.from_cfg import get_model_from_config
from trackmania_gym.trackmania_env.envs.sec_env import CrashProofEnvironment

run = None
run_path_hydra = None
model_path = None
#cfg : TrainConfig= OmegaConf.load(os.path.join("configs", "train.yaml"))

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": run_path_hydra,
    "config_name": "config.yaml",
}

@hydra.main(**_HYDRA_PARAMS)
def main(cfg : TrainConfig):

    cfg = load_and_merge_platform(cfg)
    tm_env = CrashProofEnvironment(cfg)

    try:
        tm_env.init_environment()
        
        model = get_model_from_config(cfg = cfg, tm_env = tm_env, print_params= False, run_id= "xyz" ,load_model_path=model_path)
        #model.policy.features_extractor.eval()
        # this should be sufficient 
        eval_policy = model.policy.eval()
        
        terminated = False
        observations, info = tm_env.reset()
        while True:
            action, state = eval_policy.predict(observations, deterministic=True)
            observations, reward, terminated, truncated, info = tm_env.step(action)
            print(f"{reward=}")
            if terminated or truncated: 
                observations,info = tm_env.reset()

    except Exception as e:
        traceback.print_exc()

    except KeyboardInterrupt as kinterrupt:
        print("KeyboardInterrupt")

    finally:
        # Finalize training and close game all processes.
        tm_env.finalize_process(reinit=False)
        

if __name__ == "__main__":
    main()
