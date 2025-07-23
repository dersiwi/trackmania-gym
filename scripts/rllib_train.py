"""https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#dqn"""
import os, sys
import hydra
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn import DQNConfig
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # TODO : <- i don't want this here and it shouldnt have to be here!!!


from game_interaction.run_multiprocess_wrapper import start_process_and_wait_for_startsignal
from trackmania_env.envs.enivonrments import get_environment


def make_env(cfg):
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
    # Fix working directory (Hydra changes it)
    os.chdir(hydra.utils.get_original_cwd())
    
    register_env("trackmania_env", lambda env_config: make_env(cfg))
    
    config = (
        DQNConfig()
        .environment("trackmania_env")
        .training(
            replay_buffer_config={
                "type": "PrioritizedReplayBuffer",
                "capacity": 60000,
                "alpha": 0.5,
                "beta": 0.5,
            }
        )
        .env_runners(num_env_runners=1)  # <- currently cannot increase because we can only handle one trackmania instance.
    )
    
    algo = config.build()
    
    # Training loop
    for i in range(1000):
        result = algo.train()
        print(f"Iter {i}: reward {result['episode_reward_mean']}")
        if i % 50 == 0:
            algo.save("checkpoints/trackmania")
    
    algo.stop()

if __name__ == "__main__":
    main()