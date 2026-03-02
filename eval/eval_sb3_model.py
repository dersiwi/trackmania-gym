import os
import sys

from gymnasium.spaces import box

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime
import math
import warnings
from collections.abc import Callable
from multiprocessing import Lock
from typing import Any

import hydra
import omegaconf
from omegaconf import OmegaConf

import gymnasium as gym
import numpy as np

from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecMonitor,
    VecNormalize,
    is_vecenv_wrapped,
    sync_envs_normalization,
)

from configs.config import TrainConfig
from tmn_sb3.utils.from_cfg import get_model_from_config
from trackmania_env.envs.sec_env import CrashProofEnvironment
from trackmania_env.envs.vectorized import SB3Vectorized
from utils.hydra_wandb_utils import load_and_merge_platform
from tmn_sb3.simbav2.normalizers import OnPolicySimbaVecNormalize, SimbaVecNormalize

NORMALIZERS = {"sac_simbav2": SimbaVecNormalize, "ppo_simbav2": OnPolicySimbaVecNormalize}


def print_normalization_stats(env, label: str):
    print(f"\n{'=' * 15} {label} {'=' * 15}")

    if hasattr(env, "obs_rms"):
        if isinstance(env.obs_rms, dict):
            print("obs_rms (Dict-based):")
            for key, rms in env.obs_rms.items():
                # rms is a RunningMeanStd object for each specific key
                print(f"  - Key [{key}] mean: {rms.mean}")
        else:
            print(f"obs_rms mean: {env.obs_rms.mean}")

    if hasattr(env, "ret_rms"):
        print(f"ret_rms var:  {env.ret_rms.var}")

    # only relevant for simba
    if hasattr(env, "g_max"):
        print(f"g_max:        {env.g_max}")
    if hasattr(env, "g_r_max"):
        print(f"g_r_max:      {env.g_r_max}")

    print(f"{'=' * 40}\n")


def print_results_box(results):
    max_key_len = max(len(str(k)) for k in results.keys())

    lines = []
    for key, value in results.items():
        lines.append(f"{key:<{max_key_len}} : {value}")

    content_width = max(len(line) for line in lines)
    box_width = content_width + 4  # padding

    print("┌" + "─" * box_width + "┐")
    for line in lines:
        print("│ " + line.ljust(box_width - 2) + " │")
    print("└" + "─" * box_width + "┘")


def simba_sync_envs_normalization(env: VecEnv, eval_env: VecEnv) -> None:
    """
    Synchronize the normalization statistics of an eval environment and train environment
    when they are both wrapped in a `VecNormalize` wrapper.

    :param env: Training env
    :param eval\_env: Environment used for evaluation.
    """
    sync_envs_normalization(env=env, eval_env=eval_env)
    # again only relevant for simba
    if hasattr(env, "g_max"):
        eval_env.g_max = env.g_max
    if hasattr(env, "g_r_max"):
        eval_env.g_r_max = env.g_r_max


def tmnf_evaluate_policy(
    cfg: TrainConfig,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
    image_save_path: str | None = None,
    max_video_len: int = 800,
    return_episode_rewards: bool = False,
    warn: bool = True,
    model: type_aliases.PolicyPredictor | None = None,
    model_path: str | None = None,
    vec_normalize_path: str | None = None,
    train_env: gym.Env | None = None,
    use_vec_normalize: bool = False,
    render_mode: str | None = "rgb_array",
) -> dict[str, np.ndarray | str | list[int] | float]:
    """
    Evaluates the policy and optionally records the best episode video.
    Refer to ``stable_baselines3.common.evaluation.evaluate_policy`` for detailed logic.

    :param cfg: Configuration object for environment and model setup.
    :param n_eval_episodes: Number of episodes to run for evaluation.
    :param deterministic: Whether to use deterministic actions.
    :param render: If True, captures frames for video recording.
    :param callback: Function called at every step.
    :param image_save_path: Filename to save the video of the highest-reward episode.
    :param max_video_len: Maximum frames to hold in memory to prevent overflow.
    :param return_episode_rewards: If True, returns full reward lists in the result dict.
    :param warn: Whether to warn if the environment lacks a Monitor wrapper.
    :param model: Pre-instantiated SB3-compatible model/policy.
    :param model_path: Path to a .zip model if 'model' is not provided.
    :param vec_normalize_path: Path to load VecNormalize statistics.
    :param train_env: Environment to sync normalization from if no vec_normalize_path is provided.
    :param use_vec_normalize: Whether to apply observation/reward normalization.
    :return: Dictionary containing mean/std rewards, success rate, and metadata.
    """
    assert not (model_path is None and model is None), (
        "You have to either provide the path to a model (e.g. a .zip) or parse an already build model"
    )

    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor
    
    port = cfg.gmi.port
    # create environment
    if cfg.vectorized.vectorize:
        env = SB3Vectorized(
            n_envs=cfg.vectorized.n_envs,
            tracks=cfg.vectorized.tracks,
            cfg=cfg,
            obs_as_dict=True,
            step_parallel=True,
            lock=Lock(),
            render_mode=render_mode,
        )
    else:
        env = CrashProofEnvironment(cfg, render_mode=render_mode, port=cfg.gmi.port)
        env.init_environment()
        port = env.port
    env = Monitor(env)

    # sb3 needs these wrappers TODO: we need to apply here the different vecnormalizers
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    if use_vec_normalize:
        # NOTE: This goes only well if we always want to normalise every obs key
        # but for now its ok since we indead normalise every obs
        obs_keys = None
        if isinstance(env.observation_space, gym.spaces.Dict):
            obs_keys = env.observation_space.spaces.keys()

        normalizer_class: VecNormalize = NORMALIZERS.get(cfg.policy.name, VecNormalize)
        if vec_normalize_path:
            env = normalizer_class.load(load_path=vec_normalize_path, venv=env)
        else:
            assert train_env is not None
            env = normalizer_class(env, norm_obs_keys=obs_keys)
            print_normalization_stats(env, "BEFORE SYNC")
            simba_sync_envs_normalization(env=train_env, eval_env=env)
            print_normalization_stats(env, "AFTER SYNC")

        assert isinstance(env, VecNormalize)

    env.training = False

    assert env.render_mode == render_mode

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    if model is None:
        model = get_model_from_config(
            cfg=cfg,
            tm_env=env,
            print_params=True,
            run_id=" ",
            load_model_path=model_path,
        )

    n_envs = env.num_envs

    episode_rewards = []  # This is the reward that the algos get to see (the modified one e.g. normed)
    episode_total_rewards = []  # Unmodified ground truth
    episode_lengths = []
    race_finished_binary = []
    steps_taken_finish = []

    # Video related things
    best_total_reward = -float("inf")  # we only want to save the video of the best run
    best_run_frames: list[np.ndarray] = []
    current_frames: list[np.ndarray] = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_total_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)

        if render:
            frame: np.ndarray = env.render()
            assert frame is not None
            current_frames.append(frame)

        current_rewards += rewards
        for i in range(n_envs):
            total_r = infos[i]["rewards"]["total"]
            current_total_rewards[i] += total_r
        current_lengths += 1

        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1

                    # we only want to save the video of the best run
                    if current_total_rewards[i] > best_total_reward:
                        best_total_reward = current_total_rewards[i]
                        ep_len = current_lengths[i]
                        best_run_frames = list(current_frames[-ep_len:])
                        print(f"New Best! Reward: {best_total_reward:.2f} (Length: {ep_len})")
                    # enforce a limit such that our memory does not blow up
                    if len(current_frames) > max_video_len * 2:
                        current_frames = current_frames[-max_video_len:]

                    episode_total_rewards.append(current_total_rewards[i])
                    was_finished = infos[i].get("race_finished", False)
                    race_finished_binary.append(int(was_finished))
                    # Only track length if they actually finished the race
                    if was_finished:
                        steps_taken_finish.append(current_lengths[i])

                    current_rewards[i] = 0
                    current_total_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

    if image_save_path and len(best_run_frames) > 0:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

        # Ensure directory exists
        os.makedirs(os.path.dirname(image_save_path), exist_ok=True)

        print(f"Saving best run video ({len(best_run_frames)} frames)...")
        # Use a standard 30 FPS or pull from cfg if available
        clip = ImageSequenceClip(best_run_frames, fps=30)
        clip.write_videofile(image_save_path)

        del clip
        del best_run_frames
        del current_frames

    elif image_save_path:
        print("Ignored saving a video as there were zero frames to save.")

    results = {
        "mean_normal_reward": np.mean(episode_rewards),
        "std_normal_reward": np.std(episode_rewards),
        "mean_total_reward": np.mean(episode_total_rewards),
        "std_total_reward": np.std(episode_total_rewards),
        "success_rate": np.mean(race_finished_binary),
        "steps_taken_finish": np.mean(steps_taken_finish),
        "media_path": image_save_path,
        "lengths": episode_lengths,
        "port": port, # we return the port under which the connection has run to avoid using the initial which may have faulted
    }

    if return_episode_rewards:
        results["all_normal_rewards"] = episode_rewards
        results["all_total_rewards"] = episode_total_rewards

    print_results_box(results)

    env.close()
    return results


run_path_hydra = None
model_path = None
# when VecNormalize wrappers got used during traing you want the statistics to be also applied during inference
vec_normalize_path = None

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": run_path_hydra,
    "config_name": "config.yaml",
}


@hydra.main(**_HYDRA_PARAMS)
def main(cfg: TrainConfig):
    cfg = load_and_merge_platform(cfg)

    try:
        OmegaConf.register_new_resolver("eval", lambda s: eval(s))
    except ValueError:
        pass
    omegaconf.OmegaConf.resolve(cfg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("eval_logs", timestamp)
    image_save_path = os.path.join(log_dir, "best_run.mp4")

    n_eval_episodes = 5
    deterministic = True
    render = True
    render_mode = "rgb_array"
    callback = None
    max_video_len = 800
    return_episode_rewards = False
    warn = True

    print(f"DEBUG: Starting evaluation | Episodes: {n_eval_episodes} | Deterministic: {deterministic}")

    tmnf_evaluate_policy(
        cfg=cfg,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        render=render,
        render_mode=render_mode,
        callback=callback,
        image_save_path=image_save_path,
        max_video_len=max_video_len,
        return_episode_rewards=return_episode_rewards,
        warn=warn,
        model_path=model_path,
    )


if __name__ == "__main__":
    main()
