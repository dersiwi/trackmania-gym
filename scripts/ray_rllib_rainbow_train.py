"""https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#dqn"""
import os, sys
import hydra
from omegaconf import OmegaConf
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn import DQNConfig
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # TODO : <- i don't want this here and it shouldnt have to be here!!!


from game_interaction.run_multiprocess_wrapper import start_process_and_wait_for_startsignal
from trackmania_env.envs.enivonrments import get_environment
from torch import nn
import ray

from ray.rllib.core.rl_module.rl_module import RLModuleSpec


from neuronal_networks.ray_rllib.rainbow_dqn import TMNDQNTorchModule
from neuronal_networks.vision_encoder.conv_NNs import VisionModelSix

# Create a global lock instance that will be inherited by workers 
LOCK_FILE_PATH = "rllib_window_focus"

# NOTE ray passes its own config object which is no longer a hydra conf object
def make_env(env_config):
    cfg_dict = env_config["cfg_dict"]
    worker_index = env_config.worker_index

    # Reconstruct the Hydra DictConfig object from the dictionary
    cfg = OmegaConf.create(cfg_dict)
    cfg.gmi.port = cfg.gmi.port + worker_index

    tmi_process, control_queue, response_queue = start_process_and_wait_for_startsignal(
        cfg, cfg.rl_env.obs_manager.img_width, cfg.rl_env.obs_manager.img_height, env_config["focus_lock_file_path"]
    )
    tm_env = get_environment(cfg, control_queue, response_queue)
    return tm_env

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "../configs",
    "config_name": "train.yaml",
}

@hydra.main(**_HYDRA_PARAMS)
def main(cfg):

    os.chdir(hydra.utils.get_original_cwd())
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True) 

    register_env("trackmania_env", make_env)
    cfg_dict_for_env = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    
    config = (
        DQNConfig()
         .environment(
            "trackmania_env", 
            env_config={
                "cfg_dict": cfg_dict_for_env,
                "focus_lock_file_path": LOCK_FILE_PATH
            }
        )
        .framework("torch")
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=TMNDQNTorchModule, 
                model_config={
                    "vision_model_class": VisionModelSix,
                    "share_feature_extractor": True,
                    "extractor_per_component_dim": 64,
                    "float_model_layers": [64, 32],
                    "activation_fn": nn.ReLU,
                    "last_activation_fn": nn.Identity,
                    "normalized_image": True,
                    "vision_model_params": {},
                    "device": "cuda" ,#cfg.platforms.device,
                    "advantage_head_layers": [512,256],
                    "value_head_layers": [256],
                },
            )
        )
        .training(
            lr=0.00025,
            gamma=0.99,
            # Rainbow-specific settings
            n_step=3, # Multi-step learning
            noisy=True, # Noisy networks
            v_min=-10.0,
            v_max=10.0,
            num_atoms=51, # Distributional DQN
            replay_buffer_config={
                "type": "PrioritizedEpisodeReplayBuffer",
                "capacity": 200000,
                "alpha": 0.5,
                "beta": 0.5,
                "replay_sequence_length": 1,
            },
            double_q=True, # Double Q-learning
            dueling=True, # Dueling networks
            # train_batch_size, minibatch_size, and grad_clip are set here
            train_batch_size=1024,
            minibatch_size=512,
            grad_clip=0.5,
        )
        .env_runners(
            sample_timeout_s=None,
            num_env_runners=1,
        )
        .resources(
            num_gpus=1,
            num_gpus_per_worker = 1
        )
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
    )
    
    algo = config.build()
    
    print("Starting PPO training with custom RLModule...")
    for i in range(1000):
        result = algo.train()
        
        # Save checkpoint
        if i % 50 == 0:
            checkpoint_path = algo.save(os.path.join(os.getcwd(), "checkpoints", "trackmania"))
            #print(f"Checkpoint saved at: {checkpoint_path}")
    
    algo.stop()
    ray.shutdown() # Ensure Ray resources are properly released

if __name__ == "__main__":
    main()