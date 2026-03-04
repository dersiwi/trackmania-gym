import os
import warnings
from typing import Any

import gymnasium as gym
import numpy as np
import wandb

from stable_baselines3.common.logger import Logger

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization, VecNormalize
from stable_baselines3.common.callbacks import EventCallback, BaseCallback

from .eval_sb3_model import tmnf_evaluate_policy
from .utils import save_rms_stats


# The normal sb3 EvalCallback but with some modifications
class TMNFEvalCallback(EventCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: gym.Env | VecEnv,
        callback_on_new_best: BaseCallback | None = None,
        callback_after_eval: BaseCallback | None = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str | None = None,
        best_model_save_path: str | None = None,
        deterministic: bool = True,
        render: bool = False,
        render_mode: str | None = "rgb_array",
        verbose: int = 1,
        warn: bool = True,
        save_vecnormalize: bool = True,
        policy_name: str | None = "simbav2",
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.render_mode = render_mode
        self.warn = warn
        self.save_vecnormalize = save_vecnormalize
        self.policy_name = policy_name

        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "tmnf_evaluations")
        self.log_path = log_path

        self.evaluations_results: list[list[float]] = []  # modified rewards e.g. by wrappers
        self.evaluations_timesteps: list[int] = []
        self.evaluations_length: list[list[int]] = []
        # For computing success rate
        self._is_success_buffer: list[bool] = []
        self.evaluations_successes: list[list[bool]] = []

        self.evaluations_total_results = []  # Ground truth rewards
        self.evaluations_steps_to_finish = []

    def _init_callback(self) -> None:
        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # Init callback called on new best model
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _log_success_callback(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Reset success rate buffer
            self._is_success_buffer = []

            video_path = None
            if self.render and self.best_model_save_path:
                video_path = os.path.join(self.best_model_save_path, f"best_eval_{self.num_timesteps}.mp4")

            results = tmnf_evaluate_policy(
                env=self.eval_env,
                train_env=self.training_env,
                model=self.model,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                render_mode=self.render_mode,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                image_save_path=video_path,
                use_vec_normalize=self.save_vecnormalize,
                norm_class_name=self.policy_name,
            )
            self.logger.record("tmnf_eval/mean_undis_mod_return", float(results["mean_undis_mod_return"]))
            self.logger.record("tmnf_eval/std_undis_mod_return", float(results["std_undis_mod_return"]))
            self.logger.record("tmnf_eval/mean_undis_raw_return", float(results["mean_undis_raw_return"]))
            self.logger.record("tmnf_eval/std_undis_raw_return", float(results["std_undis_raw_return"]))
            self.logger.record("tmnf_eval/success_rate", float(results["success_rate"]))

            if float(results["success_rate"]) > 0:
                # NOTE: when doing more episodes per eval one would want to check succes_rate > 0 since we could
                # have anything in [0,1] as succesrate then
                steps_finish = results["steps_taken_finish"]
                self.logger.record(
                    "tmnf_eval/steps_taken_finish", float(np.mean(steps_finish) if np.iterable(steps_finish) else steps_finish)
                )

            self.logger.record("tmnf_eval/mean_ep_length", np.mean(results["lengths"]))

            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(results.get("all_normal_rewards", []))
            self.evaluations_total_results.append(results.get("all_total_rewards", []))
            self.evaluations_length.append(results["lengths"])
            self.evaluations_successes.append(results["success_rate"])

            if self.log_path is not None:
                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    total_results=self.evaluations_total_results,
                    ep_lengths=self.evaluations_length,
                    success_rate=self.evaluations_successes,
                )

            # if there are any videos log them directly during training and do not wait till training has finished
            if video_path and os.path.exists(video_path):
                video_file = os.path.basename(video_path)
                print(f"Pushing live video to W&B: {video_file}")

                wandb.log(
                    {"tmnf_eval/video": wandb.Video(video_path, caption=f"Best Model Eval @ {self.num_timesteps} steps")},
                    step=self.num_timesteps,
                )

            # Best Model Logic
            if results["mean_undis_mod_return"] > self.best_mean_reward:
                if self.verbose >= 1:
                    print(f"New best reward: {results['mean_undis_mod_return']:.2f}")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                    if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
                        # Save the VecNormalize statistics
                        vec_normalize_path = os.path.join(self.best_model_save_path, "best_vecnormalize.pkl")
                        vec_normalizer = self.model.get_vec_normalize_env()
                        save_rms_stats(vec_normalizer, vec_normalize_path)
                        if self.verbose >= 1:
                            print(f"Saving VecNormalize stats to {vec_normalize_path}")
                self.best_mean_reward = float(results["mean_undis_mod_return"])

            self.logger.dump(self.num_timesteps)
            if self.callback:
                self._on_event()

        return continue_training

    def update_child_locals(self, locals_: dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        if self.callback:
            self.callback.update_locals(locals_)


class TMNFCheckpointCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.
    By default, it only saves model checkpoints,
    you need to pass ``save_replay_buffer=True``,
    and ``save_vecnormalize=True`` to also save replay buffer checkpoints
    and normalization statistics checkpoints.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq: Save checkpoints every ``save_freq`` call of the callback.
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param save_replay_buffer: Save the model replay buffer
    :param save_vecnormalize: Save the ``VecNormalize`` statistics
    :param verbose: Verbosity level: 0 for no output, 2 for indicating when saving model checkpoint
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self.save_path, f"{self.name_prefix}_{checkpoint_type}{self.num_timesteps}_steps.{extension}")

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = self._checkpoint_path(extension="zip")
            self.model.save(model_path)
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")

            if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                # If model has a replay buffer, save it too
                replay_buffer_path = self._checkpoint_path("replay_buffer_", extension="pkl")
                self.model.save_replay_buffer(replay_buffer_path)  # type: ignore[attr-defined]
                if self.verbose > 1:
                    print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

            if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
                # Save the VecNormalize statistics
                vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl")
                vec_normalizer = self.model.get_vec_normalize_env()
                save_rms_stats(vec_normalizer, vec_normalize_path)
                if self.verbose >= 2:
                    print(f"Saving model VecNormalize to {vec_normalize_path}")

        return True
