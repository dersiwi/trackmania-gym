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
from ray.rllib.algorithms.ppo import PPOConfig 
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from neuronal_networks.ray_rllib.modules import TMNFActorCriticModule
from neuronal_networks.vision_encoder.conv_NNs import VisionModelSix

# NOTE ray passes its own config object which is no longer a hydra conf object
def make_env(env_config):
    cfg_dict = env_config["cfg_dict"]
    
    # Reconstruct the Hydra DictConfig object from the dictionary
    cfg = OmegaConf.create(cfg_dict)
    tmi_process, control_queue, response_queue = start_process_and_wait_for_startsignal(
        cfg, cfg.rl_env.obs_manager.img_width, cfg.rl_env.obs_manager.img_height
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
        PPOConfig()
         .environment(
            "trackmania_env", 
            env_config={
                "cfg_dict": cfg_dict_for_env,
            }
        )
        .framework("torch")
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=TMNFActorCriticModule, 
                model_config={
                    "vision_model_class": VisionModelSix,
                    "share_feature_extractor": True,
                    "extractor_per_component_dim": 64,
                    "float_model_layers": [64, 32],
                    "activation_fn": nn.ReLU,
                    "last_activation_fn": nn.Identity,
                    "normalized_image": True,
                    "vision_model_params": {},
                    "device": "cpu"#cfg.platforms.device,
                },
            )
        )
        .training(
            lr=0.00025,
            gamma=0.99,  
            lambda_=0.95, 
            kl_coeff=0.2, 
            clip_param=0.2, 
            vf_loss_coeff=1.0, 
            entropy_coeff=0.01,
            num_sgd_iter=5, 
            train_batch_size=2048,
            grad_clip=0.5,
        )
        .env_runners(
            num_env_runners=1,
        )
        .resources(
            num_gpus=0
        )
    )
    
    algo = config.build()
    
    print("Starting PPO training with custom RLModule...")
    for i in range(1000):
        result = algo.train()
        print(f"Iter {i}: reward {result['episode_reward_mean']:.2f}, episodes_this_iter={result['episodes_this_iter']}")
        
        # Save checkpoint
        if i % 50 == 0:
            checkpoint_path = algo.save(os.path.join(os.getcwd(), "checkpoints", "trackmania"))
            print(f"Checkpoint saved at: {checkpoint_path}")
    
    algo.stop()
    ray.shutdown() # Ensure Ray resources are properly released

if __name__ == "__main__":
    main()