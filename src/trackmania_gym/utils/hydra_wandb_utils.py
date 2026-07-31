
import wandb
from wandb.wandb_run import Run

from omegaconf import OmegaConf, open_dict 
from omegaconf.errors import ConfigAttributeError

from configs.config import TrainConfig
from typing import Callable

class RuntimeCfg:

    cfg : TrainConfig = None
    """This is the global config variable, use this sparingly and only when absolutely necessary. """

    @staticmethod
    def setcfg(cfg : TrainConfig) -> None:
        RuntimeCfg.cfg = cfg

def secure_attribute_retrieval(getter : Callable, default : any = None):
    """Securely returns an attribute from the config.
    Args:
        getter (Callable)   : Lambda function retrieving attribute from config : `lambda: cfg.gmi` (e.g.)
        default (any)       : Default value to be returned if getter fails
    """
    try:    
        return getter()
    except ConfigAttributeError:
        return default


def init_and_login_wandb(cfg : TrainConfig, wandbdir : str = "wandb",run_id = None, resume = None ) -> tuple[Run | None, str]:
    """Instanciates and logs into weights and biases (wandb), if specified in configuration (cfg.wandb.use).
    After login, returns tuple of Run-instance and run-id 
    
    If cfg.wandb.use is False, the returned Run is None.

    :param cfg: The configuration object containing wandb settings.
    :param wandbdir: The directory where wandb run data will be stored.
    :param run_id: The ID of a previous run to resume.
    :param resume: The resume behavior for the wandb run (e.g., "allow", "must", "never")
    """
    #run_id = run_id or ""
    if cfg.wandb.use:
        wandb.login()
        wandb_conf = OmegaConf.to_container(cfg, resolve=True,throw_on_missing=True)
        run = wandb.init(
            entity=cfg.wandb.entity, 
            project=cfg.wandb.project,
            sync_tensorboard=True, 
            monitor_gym=True,  
            save_code=True,
            dir = wandbdir,
            config=wandb_conf,
            id = run_id,
            resume= resume)
        run_id = run.id
        wandb.config.update(wandb_conf,allow_val_change=True) # if run gets resumed then this has to updated and not overriden via wandb.config = wandbn_conf
        return run, run_id
    else:
        return None, ""
    
from hydra.utils import to_absolute_path

def load_and_merge_yaml(cfg : TrainConfig, yaml_to_merge : str) -> TrainConfig:
    """Loads a yaml and merges it with cfg
    Args:
        cfg (TrainConfig)   : Configuration provided by hydra
        yaml_to_merge (str) : Path to external yaml file to merge with yaml
    Returns:
        Merged (TrainConcig): Merged hydra-config with external yaml."""
    yaml = OmegaConf.load(to_absolute_path(yaml_to_merge))
    with open_dict(cfg):
        merged = OmegaConf.merge(cfg, yaml)
    return merged



def load_and_merge_platform(cfg : TrainConfig) -> TrainConfig:
    """Load and merge the standalone platform-yaml and merge it into the global 'cfg'-yaml. Also sets RuntimeCfg."""
    platform_path = None
    platform_default_path = "configs/platforms.yaml"
    try:
        platform_path = cfg.platforms_config_path
    except ConfigAttributeError as e:
        print(f"[!!] Platform-path attribute missing in cfg; config too old? Trying default path : {platform_default_path}")
        platform_path = platform_default_path
    assert not platform_path is None
    merged_cfg = load_and_merge_yaml(cfg, platform_path)
    RuntimeCfg.setcfg(merged_cfg)
    return merged_cfg
