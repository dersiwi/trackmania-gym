
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class GMIConfig:
    launch: bool
    tm_loader_profile_name: str
    headless: bool
    port: int

@dataclass
class ImageConfig:
    width: int
    height: int

@dataclass
class PlatformConfig:
    os: str
    home: str
    tmloader: str
    plugin: str

@dataclass
class PretrainedResNetConfig:
    _partial_: bool
    _target_: str
    model_name: str
    out_dims: int
    pretrained: bool
    trainable_backbone: bool

@dataclass
class ObservationsFilterConfig:
    _partial_: bool
    _target_: str
    observations_list: List[str]

@dataclass
class BGRAtoRGBConfig:
    _partial_: bool
    _target_: str

@dataclass
class TransformGrayscaleConfig:
    _partial_: bool
    _target_: str
    keep_dim: bool

@dataclass
class PytorchWrapperConfig:
    _partial_: bool
    _target_: str

@dataclass
class WrappersConfig:
    observations_filter: ObservationsFilterConfig
    bgra_to_rgb: BGRAtoRGBConfig
    transform_grayscale: TransformGrayscaleConfig
    pytroch_wrapper: PytorchWrapperConfig

@dataclass
class RLEnvConfig:
    wrappers: WrappersConfig

@dataclass
class SB3PPOConstructorConfig:
    _partial_: bool
    _target_: str
    policy: str
    verbose: int
    n_steps : int

@dataclass
class SB3PPOLearnArgsConfig:
  total_timesteps: int
  log_interval: int
  tb_log_name: str
  reset_num_timesteps: bool
  progress_bar: bool
@dataclass
class SB3PPOConfig:
    constructor: SB3PPOConstructorConfig
    learn_args: SB3PPOLearnArgsConfig

@dataclass
class SB3CallbackConfig:
    _partial_: bool
    _target_: str
    model_save_freq: int
    gradient_save_freq: int
    log: str
    verbose: int

@dataclass
class WandbConfig:
    use: bool
    entity: str
    project: str

@dataclass
class TrainConfig:
    gmi: GMIConfig
    image: ImageConfig
    total_timesteps: int
    platforms: Optional[PlatformConfig] = None
    models: Optional[PretrainedResNetConfig] = None
    rl_env: Optional[RLEnvConfig] = None
    sb3: Optional[SB3PPOConfig] = None
    wandb_callbacks: Optional[SB3CallbackConfig]= None
    wandb: Optional[WandbConfig] = None