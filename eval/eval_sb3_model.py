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
    is_vecenv_wrapped,
)

from configs.config import TrainConfig
from tmn_sb3.utils.from_cfg import get_model_from_config
from trackmania_env.envs.sec_env import CrashProofEnvironment
from trackmania_env.envs.vectorized import SB3Vectorized
from utils.hydra_wandb_utils import (
    load_and_merge_platform,
    secure_attribute_retrieval,
)


def print_results_box(results):
    # Determine the longest key for alignment
    max_key_len = max(len(str(k)) for k in results.keys())

    # Build formatted lines
    lines = []
    for key, value in results.items():
        lines.append(f"{key:<{max_key_len}} : {value}")

    # Determine box width
    content_width = max(len(line) for line in lines)
    box_width = content_width + 4  # padding

    # Print top border
    print("┌" + "─" * box_width + "┐")

    # Print content
    for line in lines:
        print("│ " + line.ljust(content_width) + " │")

    # Print bottom border
    print("└" + "─" * box_width + "┘")


def tmnf_evaluate_policy(
    cfg: TrainConfig,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
    image_save_path: str | None = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
    model: type_aliases.PolicyPredictor | None = None,
    model_path: str | None = None,
) -> dict[str, np.ndarray | str | list[int] | float]:
    """
    Runs the policy for ``n_eval_episodes`` episodes and outputs the average return
    per episode (sum of undiscounted rewards).
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a ``predict`` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to perform additional checks,
        called ``n_envs`` times after each step.
        Gets locals() and globals() passed as parameters.
        See https://github.com/DLR-RM/stable-baselines3/issues/1912 for more details.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean return per episode (sum of rewards), std of reward per episode.
        Returns (list[float], list[int]) when ``return_episode_rewards`` is True, first
        list containing per-episode return and second containing per-episode lengths
        (in number of steps).
    """
    assert not (model_path is None and model is None), (
        "You have to either provide the path to a model (e.g. a .zip) or parse an already build model"
    )

    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if cfg.vectorized.vectorize:
        env = SB3Vectorized(
            n_envs=cfg.vectorized.n_envs,
            tracks=cfg.vectorized.tracks,
            cfg=cfg,
            obs_as_dict=True,
            step_parallel=True,
            lock=Lock(),
        )
    else:
        env = CrashProofEnvironment(cfg)
        env.init_environment()

    env = Monitor(env)

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

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

        if render:
            env.render()

    results = {
        "mean_normal_reward": np.mean(episode_rewards),
        "std_normal_reward": np.std(episode_rewards),
        "mean_total_reward": np.mean(episode_total_rewards),
        "std_total_reward": np.std(episode_total_rewards),
        "success_rate": np.mean(race_finished_binary),
        "media_path": saved_media_path,
        "lengths": episode_lengths,
    }

    if return_episode_rewards:
        results["all_normal_rewards"] = episode_rewards
        results["all_total_rewards"] = episode_total_rewards

    print_results_box(results)

    return results


run_path_hydra = None
model_path = None

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

    n_eval_episodes = 10
    deterministic = True
    render = False
    callback = None
    image_save_path = None
    return_episode_rewards = False
    warn = True
    model = None  # If None, get_model_from_config will load it from model_path

    print(f"DEBUG: Starting evaluation | Episodes: {n_eval_episodes} | Deterministic: {deterministic}")

    tmnf_evaluate_policy(
        cfg=cfg,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        render=render,
        callback=callback,
        image_save_path=image_save_path,
        return_episode_rewards=return_episode_rewards,
        warn=warn,
        model=model,
    )


if __name__ == "__main__":
    main()
