import pickle
import os
import numpy as np
from typing import Any


def save_rms_stats(env: Any, save_path: str):
    """
    Extracts mean, var, and count from VecNormalize (including SimBa variants)
    and saves them to a file.
    """
    stats = {
        "obs": {},
        "ret": {"mean": np.copy(env.ret_rms.mean), "var": np.copy(env.ret_rms.var), "count": float(env.ret_rms.count)},
    }

    if isinstance(env.obs_rms, dict):
        for key, rms in env.obs_rms.items():
            stats["obs"][key] = {"mean": np.copy(rms.mean), "var": np.copy(rms.var), "count": float(rms.count)}
    else:
        stats["obs"] = {"mean": np.copy(env.obs_rms.mean), "var": np.copy(env.obs_rms.var), "count": float(env.obs_rms.count)}

    if hasattr(env, "g_r_max"):
        stats["g_r_max"] = float(env.g_r_max)
    if hasattr(env, "g_max"):
        stats["g_max"] = float(env.g_max)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(stats, f)

    print(f"Successfully saved normalization stats to: {save_path}")


def load_rms_stats(env: Any, load_path: str):
    """
    Loads mean, var, and count from a file and injects them into the environment.
    """
    with open(load_path, "rb") as f:
        stats = pickle.load(f)

    if not hasattr(env, "ret_rms"):
        raise AttributeError("Environment is missing 'ret_rms' but stats file has it.")

    env.ret_rms.mean = np.copy(stats["ret"]["mean"])
    env.ret_rms.var = np.copy(stats["ret"]["var"])
    env.ret_rms.count = stats["ret"]["count"]

    if isinstance(env.obs_rms, dict):
        if not isinstance(stats["obs"], dict):
            raise TypeError("Env expects dict-based obs_rms.")

        file_keys = set(stats["obs"].keys())
        env_keys = set(env.obs_rms.keys())
        if file_keys != env_keys:
            raise KeyError(f"Mismatch in observation keys! File: {file_keys}, Env: {env_keys}")

        for key in env_keys:
            env.obs_rms[key].mean = np.copy(stats["obs"][key]["mean"])
            env.obs_rms[key].var = np.copy(stats["obs"][key]["var"])
            env.obs_rms[key].count = stats["obs"][key]["count"]
    else:
        if isinstance(stats["obs"], dict):
            raise TypeError("Env expects single array obs_rms, but file contains dict-based stats.")

        env.obs_rms.mean = np.copy(stats["obs"]["mean"])
        env.obs_rms.var = np.copy(stats["obs"]["var"])
        env.obs_rms.count = stats["obs"]["count"]

    simba_attrs = ["g_r_max", "g_max"]
    for attr in simba_attrs:
        in_file = attr in stats
        in_env = hasattr(env, attr)

        if in_file and not in_env:
            raise AttributeError(f"Stats file contains '{attr}', but the environment class does not support it.")
        if in_env and not in_file:
            raise AttributeError(f"Environment expects '{attr}', but it was missing from the stats file.")

        if in_file and in_env:
            setattr(env, attr, stats[attr])

    print(f"Successfully loaded and validated normalization stats from: {load_path}")
