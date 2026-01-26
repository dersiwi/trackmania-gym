
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class GMIConfig:
    launch: bool
    tm_loader_profile_name: str
    headless: bool
    port: int
    track : str
    reference_line : str
    camera: int


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
class ModelCfg:
    _partial_: bool
    _target_: str
    model_name: str
    out_dims: int
    args : dict[str, any]
    """Argruments given to model at instanciation"""

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
class ObservationManagerConfig:
    """ObservationManagerConfig may contain attributes custom to a specific observation manager. These only list the common ones between all observation managers."""
    colorspace : str
    convert_torch : bool
    img_height : int
    img_width : int
    name : str



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
    eval_freq : int
    logminmax_continuous : bool
    
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
    wrap_obs_in_test : bool
    n_previous_actions : int
    camera_id : int

    normalize_rewards : bool
    normalize_obs : bool
   
    """Executes a check of number of channels in order to determine if a observation-term is an imageterm.
    May be wrong, if doing framestacking,..."""

    continuous_actions : bool
    actiondim : int

    ignore_stuck_for_n_steps_after_reset : int
    increase_timeout_intervals : list[int]
    new_timeouts : list[int]
    terminate_after_steps_without_progress : int

    actions_per_second : int
    automatic_prevent_sim_finish : bool
    disable_waitforstep_after_n_consecutive_timeouts : int
    waitforstep_timeout_in_s : float
    use_rewind : bool

    startposition_accuracy_threshold : float


@dataclass
class RewardManagerCfg:
    args : any
    name : str

@dataclass
class TerminationManagerCfg:
    """Specific termination managers may have more attributes than listed here, this only lists the common ones."""
    max_steps_until_reset : int
    name : str
    ignore_stuck_for_n_steps_after_reset : int 
    
@dataclass
class RLEnvConfig:
    wrappers: WrappersConfig
    env : EnvConfig
    obs_manager : ObservationManagerConfig
    linesightobsmanager : LinesightObsCfg
    reward_manager : RewardManagerCfg
    termination_manager : TerminationManagerCfg

@dataclass
class LearningRateSchedulerConfig:
    initial_value : float
@dataclass
class Learnargs:
    total_timesteps : int
    log_interval : int
    reset_num_timesteps : bool
    progress_bar : bool
    tb_log_name : str


@dataclass
class PolicyCfg:
    name : str
    type : str
    extractors_out_dim : Optional[int] 
    """Custom policies may vary."""
@dataclass
class FloatNetArchitecture:
    extractors_out_dim : int
    observations : list[str]
    float_net : list[int]
    activation_fn : any
    """this is a target-path to a function that is being instanciated"""
    last_activation_fn : any
    """this is a target-path to a function that is being instanciated"""

@dataclass
class ActorCriticPolicyCfg(PolicyCfg):
    actor : FloatNetArchitecture
    critic : FloatNetArchitecture

@dataclass
class TrainConfig:
    gmi: GMIConfig
    image: ImageConfig
    learn_args : Learnargs
    extractors_out_dim : int
    platforms_config_path : str
    debug: bool
    img_dump_freq:int
    img_dump_path:int | None
    headless: bool
    policy : PolicyCfg | ActorCriticPolicyCfg
    platforms: PlatformConfig 
    models: ModelCfg 
    rl_env: RLEnvConfig 
    lr_scheduler : LearningRateSchedulerConfig 
    sb3: SB3PPOConfig 
    wandb_callbacks: SB3CallbackConfig 
    wandb: WandbConfig 

    
