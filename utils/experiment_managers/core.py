import os
import glob
import wandb
from typing import Any, List
from dataclasses import dataclass
from configs.config import TrainConfig
from utils.hydra_wandb_utils import init_and_login_wandb

@dataclass
class LoggingWandbArtifact:
    """Simple data structure to hold artifact logging data."""
    artifact_name: str
    artifact_type: str
    path: str

class ExperimentManager:
    """
    Handles initialization and finalization for ML experiment runs.
    
    Responsibilities:
      - Prepare directories (models, logs, checkpoints)
      - Initialize external tools (e.g. W&B, Hydra)
      - Handle post-training artifact upload and cleanup
    """
    
    def __init__(self, hydra_run_dir: str, cfg: TrainConfig, run_id=None, resume=None):
        self.hydra_run_dir = hydra_run_dir
        self.cfg = cfg
        self.resume = resume
        self.run, self.run_id = init_and_login_wandb(
            self.cfg, 
            wandbdir=self.hydra_run_dir,
            run_id=run_id,
            resume=self.resume
        )
        self.run_id_in_hydra_log_dir = os.path.join(self.hydra_run_dir, self.run_id)
        
        self.artifacts_to_log: List[LoggingWandbArtifact] = []
    
    def add_artifact(self, artifact_name: str, artifact_type: str, path: str):
        """
        Add an artifact (file or directory) to be uploaded to W&B
        at the end of the training run.
        """
        art = LoggingWandbArtifact(
            artifact_name=artifact_name,
            artifact_type=artifact_type,
            path=path
        )
        self.artifacts_to_log.append(art)
        print(f"Added artifact '{artifact_name}' for upload from {path}")

    def _additional_post_processing(self,model: Any):
        pass

    def after_training(self, model: Any = None):
        """Save final model, upload all queued artifacts, and close loggers."""
        if self.run is None:
            print("W&B disabled, skipping post-training artifact logging.")
            return
            
        print("Running post-training tasks...")
       
        self._additional_post_processing(model)

        # per default always log the hydra conf 
        hydra_dir_path = os.path.join(self.hydra_run_dir, ".hydra")
        self.add_artifact(
            artifact_name="hydra",
            artifact_type="hydra_conf",
            path=hydra_dir_path
        )
        
        self._upload_artifacts()
        self._finalize_wandb()

    def _upload_artifacts(self):
        """
        Logs all queued artifacts to W&B 
        """
        if self.run is None:
            print("W&B run not initialized, skipping artifact upload.")
            return

        print(f"Uploading {len(self.artifacts_to_log)} artifacts to W&B...")
        
        for art in self.artifacts_to_log:
            self._log_wandb_artifact(
                artifact_name=art.artifact_name,
                artifact_type=art.artifact_type,
                path=art.path
            )
        print("Artifact upload complete.")

    def _log_wandb_artifact(self, artifact_name: str, artifact_type: str, path: str):
        """
        Checks if a path exists, creates a wandb artifact,
        and logs it (differentiating between file and directory).
        """
        if not os.path.exists(path):
            print(f"Warning: Path not found, not logging artifact '{artifact_name}': {path}")
            return

        try:
            artifact = wandb.Artifact(artifact_name, type=artifact_type)
            
            if os.path.isfile(path):
                artifact.add_file(path)
            elif os.path.isdir(path):
                artifact.add_dir(path)
            else:
                print(f"Warning: Path '{path}' is not a file or directory, skipping.")
                return
                
            self.run.log_artifact(artifact)
            print(f"Successfully logged artifact '{artifact_name}' from {path}")
            
        except Exception as e:
            print(f"Error logging artifact '{artifact_name}' from path {path}: {e}")

    def _finalize_wandb(self):
        """(Example Stub) Finishes the W&B run."""
        if self.run:
            print("Finishing W&B run...")
            self.run.finish()

    def get_tensorboard_login_identifier(self) -> str:
        """identifier for the run which gets used for tensorboard login"""
        return self.run_id_in_hydra_log_dir
