
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class GMIConfig:
    launch: bool
    tm_loader_profile_name: str
    headless: bool
    port: int
    track : str
    automatic_prevent_sim_finish : bool
    reference_line : str


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
    map_dir : str
    device: str

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
class ObservationManager:
    observation_list : list[str]
    colorspace : str
    convert_torch : bool



@dataclass
class SB3PPOConstructorConfig:
    _partial_: bool
    _target_: str
    policy: str
    verbose: int
    n_steps : int

@dataclass
class LinesightObsCfg:
    n_zone_centers_in_inputs : int
    one_every_n_zone_centers_in_inputs : int
    n_zone_centers_extrapolate_after_end_of_map : int
    n_zone_centers_extrapolate_before_start_of_map : int
    n_prev_actions_in_inputs : int
    margin_to_announce_finish_meters : int
    distance_between_checkpoints: float
    road_width: int  ## a little bit of margin, could be closer to 24 probably ? Don't take risks there are curvy roads
    sync_virtual_and_real_checkpoints : bool

@dataclass
class SB3LearnArgsConfig:
  total_timesteps: int
  log_interval: int
  tb_log_name: str
  reset_num_timesteps: bool
  progress_bar: bool
  
@dataclass
class SB3PPOConfig:
    constructor: SB3PPOConstructorConfig
    algorithm_params : dict[str, any]

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
    checkpoint_freq: int 
    eval_freq:int
    
@dataclass
class EnvConfig:
    position_buffer_size : int
    position_moved_threshold : float
    reset_mode : str
    reward_calculator : str
    termination_manager : str
    obs_manager : str ##use basic for other
    max_steps_until_reset : int
    game_speed : float
    countdown_speed : float
    test : bool
    n_previous_actions : int
    ignore_stuck_for_n_steps_after_reset : int
    increase_timeout_intervals : list[int]
    new_timeouts : list[int]
    terminate_after_steps_without_progress : int


@dataclass
class RewardManagerCfg:
    rewardterm_weights : dict[str, float]
    lateral_distance_mode : str
    
@dataclass
class RLEnvConfig:
    wrappers: WrappersConfig
    env : EnvConfig
    obsmanager : ObservationManager
    linesightobsmanager : LinesightObsCfg
    reward_manager : RewardManagerCfg



@dataclass
class TrainConfig:
    gmi: GMIConfig
    image: ImageConfig
    total_timesteps: int
    extractors_out_dim : int
    learn_args: SB3LearnArgsConfig
    platforms: Optional[PlatformConfig] = None
    models: Optional[PretrainedResNetConfig] = None
    rl_env: Optional[RLEnvConfig] = None
    sb3: Optional[SB3PPOConfig] = None
    wandb_callbacks: Optional[SB3CallbackConfig]= None
    wandb: Optional[WandbConfig] = None
