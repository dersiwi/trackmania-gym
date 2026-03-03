import os
from typing import List
from stable_baselines3.common.callbacks import EventCallback, CallbackList, EvalCallback, CheckpointCallback

from stable_baselines3.common.base_class import BaseAlgorithm
from gymnasium import Env
from utils.experiment_managers.core import ExperimentManager
from configs.config import TrainConfig
from trackmania_env.callbacks import *
from eval.eval_sb3_callback import TMNFEvalCallback, TMNFCheckpointCallback

from omegaconf import OmegaConf


class Sb3ExperimentManager(ExperimentManager):
    """
    A SB3 Manager that automatically configures and registers:
    Callbacks:
        1. EvalCallback (for best model)
        2. CheckpointCallback (for periodic checkpoints)
    Artifacts:
        1. best-model
        2. checkpoint-model
    """

    def __init__(
        self,
        hydra_run_dir: str,
        cfg: TrainConfig,
        env: Env,
        eval_freq: int,
        checkpoint_freq: int,
        run_id=None,
        resume=None,
        n_envs: int = 1,
    ):
        super().__init__(hydra_run_dir, cfg, run_id, resume)
        self.callbacks: List[EventCallback] = []
        model_dir = os.path.join(self.hydra_run_dir, "models")
        self.best_model_path = self._setup_path(model_dir, "best_model")
        self.checkpoint_path = self._setup_path(model_dir, "checkpoints")
        self.eval_log_path = self._setup_path(self.hydra_run_dir, "eval_logs")

        self.eval_freq = eval_freq
        self.checkpoint_freq = checkpoint_freq
        self.n_envs = n_envs

        self.cfg = cfg
        self.cfg.gmi.port *= 2 # to prevent port clashes


        self.save_vecnormalize = self.cfg.eval.save_vecnormalize or (cfg.policy.type == "simbav2") #NOTE: hard coded but got no time

        self.append_callbacks(env)
        self.add_artifacts()

        

    def get_callbacks(self) -> CallbackList:
        """
        Returns all registered callbacks wrapped in an SB3-native CallbackList.
        """
        return CallbackList(self.callbacks)

    def append_callbacks(self, env):
        """Append Evaluation-callback and Checkpoint-Callback"""
        # Eval Callback saves best model
        eval_callback = TMNFEvalCallback(
            cfg=self.cfg,
            n_eval_episodes=self.cfg.eval.n_eval_episodes,
            eval_freq=self.cfg.eval.eval_freq,
            best_model_save_path=self.best_model_path,
            log_path=os.path.join(self.hydra_run_dir, "eval_logs"),
            deterministic=self.cfg.eval.deterministic,
            render= self.cfg.eval.render,
            render_mode= self.cfg.eval.render_mode,
            verbose= self.cfg.eval.verbose,
            warn=self.cfg.eval.warn,
            save_vecnormalize= self.save_vecnormalize,
        )
        # eval_callback = EvalCallback(
        #     env,
        #     best_model_save_path=self.best_model_path,
        #     log_path=os.path.join(self.hydra_run_dir, "eval_logs"),
        #     eval_freq=self.eval_freq,
        #     deterministic=True,
        #     render=False,
        # )
        self.callbacks.append(eval_callback)

        # Checkpoint Callback save model every N steps
        checkpoint_callback = TMNFCheckpointCallback(
            save_freq=self.checkpoint_freq,
            save_path=self.checkpoint_path,
            name_prefix="checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=self.save_vecnormalize,
        )
        self.callbacks.append(checkpoint_callback)

        if self.use_wandb:
            self.callbacks.append(AccumRewardLogCallback(n_envs=self.n_envs))
            self.callbacks.append(ReturnCallback())
            self.callbacks.append(FurtherStatisticsCallback())
            self.callbacks.append(BinaryRaceFinished())


            if self.cfg.rl_env.env.continuous_actions:
                self.callbacks.append(ContinuousActionLogCallback(log_minmax=self.cfg.wandb.logminmax_continuous))

        if self.cfg.debug and OmegaConf.select(cfg=self.cfg, key="rl_env.obs_manager.obs_have_imgs", default=False):
            self.callbacks.append(
                ImageDumpCallback(verbose=0, dump_freq=self.cfg.img_dump_freq, dump_dir=self.cfg.img_dump_path)
            )

    def add_artifacts(self):
        """Add best model and checkpoint-model artifact"""
        best_model_zip_path = os.path.join(
            self.best_model_path, "best_model.zip"
        )  # Register the best_model.zip file that EvalCallback will create
        self.add_artifact(artifact_name="best_model", artifact_type="model", path=best_model_zip_path)

        # Register all checkpoint files using a glob pattern
        self.add_artifact(artifact_name="checkpoint_model", artifact_type="model", path=self.checkpoint_path)

    def _setup_path(self, base_dir: str, sub_dir: str) -> str:
        path = os.path.join(base_dir, sub_dir)
        os.makedirs(path, exist_ok=True)
        return path

    def _additional_post_processing(self, model: BaseAlgorithm):
        final_model_path = os.path.join(self.hydra_run_dir, "model.zip")
        model_save_base = os.path.join(self.hydra_run_dir, "model")

        try:
            print(f"Saving final model to {model_save_base}...")
            model.save(model_save_base)

            self.add_artifact(artifact_name="final_model", artifact_type="model", path=final_model_path)
        except Exception as e:
            print(f"Error saving final model: {e}")
