import os
import warnings
from typing import Any

import gymnasium as gym
import numpy as np

from stable_baselines3.common.logger import Logger

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common.callbacks import EventCallback

from .eval_sb3_model import tmnf_evaluate_policy


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
        cfg,
        callback_on_new_best: BaseCallback | None = None,
        callback_after_eval: BaseCallback | None = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str | None = None,
        best_model_save_path: str | None = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        save_vecnormalize: bool = True,
    ):
        super().__init__(callback_after_eval, verbose=verbose)

        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            # Give access to the parent
            self.callback_on_new_best.parent = self

        self.cfg = cfg
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn
        self.save_vecnormalize = save_vecnormalize

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
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            video_path = None
            if self.render and self.best_model_save_path:
                video_path = os.path.join(self.best_model_save_path, f"best_eval_{self.num_timesteps}.mp4")

            results = tmnf_evaluate_policy(
                cfg=self.cfg,
                train_env=self.training_env,
                model=self.model,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                image_save_path=video_path,
            )
            self.logger.record("tmnf_eval/mean_reward", float(results["mean_normal_reward"]))
            self.logger.record("tmnf_eval/std_reward", float(results["std_normal_reward"]))
            self.logger.record("tmnf_eval/mean_total_reward", float(results["mean_total_reward"]))
            self.logger.record("tmnf_eval/std_total_reward", float(results["std_total_reward"]))
            self.logger.record("tmnf_eval/success_rate", float(results["success_rate"]))

            # Use .get() for steps_taken_finish in case no episodes finished
            if float(results["success_rate"]) == 1:
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

            # Best Model Logic
            if results["mean_total_reward"] > self.best_mean_reward:
                if self.verbose >= 1:
                    print(f"New best reward: {results['mean_total_reward']:.2f}")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(results["mean_total_reward"])

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
