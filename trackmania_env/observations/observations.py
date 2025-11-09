from trackmania_env.observations.observation_manager import ObservationManager
from trackmania_env.observations.observation_test import ObservationTest
 
from configs.config import TrainConfig, ObservationManagerConfig

from trackmania_env.observations.implementations.linesight_obs_wrapper import LinesightObservationManager
from trackmania_env.observations.implementations.nextpoint_obs import NextPointObsManager, VisionLessNextPointObsManager 
from trackmania_env.observations.implementations.sophy_obs import SophyObsManager
from trackmania_env.observations.implementations.dyn_nextpoint_obs import DynamicNextPointObsManager

from configs.config import TrainConfig, ObservationManagerConfig
from utils.hydra_wandb_utils import secure_attribute_retrieval


def get_observation_manager(cfg : TrainConfig, wrap_obs_in_test : bool = False, 
                            normalize : bool = False, 
                            grayscale_imgs_as_uint8 : bool = False, 
                            obs_have_imgs : bool = True) -> ObservationManager:
    """Instanciate Configuration manager according to confguration.
    Args:
        wrap_obs_in_test (bool)         : If True, Instanciates ObservationTest() with given instanciated observation wrapper
        normalize (bool)                : Normalizes observations to [0,1]
        grayscale_imgs_as_uint8 (bool)  : If True and images are grayscale, observation wrapper returns [0,255] value images as uint8, instead of [0,1] as float.
        obs_have_imgs (bool)            : If False, observation-manager only returns floatvector : default; true
    """
    
    obs_manager_cfg : ObservationManagerConfig = cfg.rl_env.obs_manager
    name = obs_manager_cfg.name 
    match name:
        case "linesight":
            obs_manager = LinesightObservationManager(
                colorspace = obs_manager_cfg.colorspace,
                img_width  = obs_manager_cfg.img_width,
                img_height = obs_manager_cfg.img_height,
                convert_torch = obs_manager_cfg.convert_torch,
                normalize     = normalize,
                ref_line_path= cfg.gmi.reference_line,
                map_path      = f"{cfg.platforms.map_dir}/{cfg.gmi.track}",
                n_zone_centers_extrapolate_after_end_of_map = obs_manager_cfg.n_zone_centers_extrapolate_after_end_of_map,
                distance_between_checkpoints                = obs_manager_cfg.distance_between_checkpoints,
                road_width                                  = obs_manager_cfg.road_width,
                n_zone_centers_in_inputs                    = obs_manager_cfg.n_zone_centers_in_inputs,
                margin_to_announce_finish_meters            = obs_manager_cfg.margin_to_announce_finish_meters,
                one_every_n_zone_centers_in_inputs          = obs_manager_cfg.one_every_n_zone_centers_in_inputs,
                n_zone_centers_extrapolate_before_start_of_map = obs_manager_cfg.n_zone_centers_extrapolate_before_start_of_map,
                n_prev_actions_in_inputs                    = obs_manager_cfg.n_prev_actions_in_inputs,
                sync_virtual_and_real_checkpoints           = obs_manager_cfg.sync_virtual_and_real_checkpoints,
            )

        case "nextpointobs":
            obs_manager = NextPointObsManager(colorspace=obs_manager_cfg.colorspace,
                                        convert_torch=obs_manager_cfg.convert_torch,
                                        img_width=obs_manager_cfg.img_width, 
                                        img_height=obs_manager_cfg.img_height,
                                        normalize = normalize)
        
        case "sophy": 
            obs_manager = SophyObsManager(colorspace=obs_manager_cfg.colorspace,
                                        convert_torch=obs_manager_cfg.convert_torch,
                                        normalize= normalize,
                                        img_width=obs_manager_cfg.img_width, 
                                        img_height=obs_manager_cfg.img_height,
                                        maxlen_history= obs_manager_cfg.maxlen_history,
                                        lookahead_sec = obs_manager_cfg.lookahead_sec,
                                        n_points = obs_manager_cfg.n_points)
        case "dyna_nextpoint":
            obs_manager = DynamicNextPointObsManager(colorspace=obs_manager_cfg.colorspace,
                                        convert_torch=obs_manager_cfg.convert_torch,
                                        img_width=obs_manager_cfg.img_width, 
                                        img_height=obs_manager_cfg.img_height,
                                        normalize = normalize,
                                        lookahead_sec =  obs_manager_cfg.lookahead_sec,
                                        n_points=  obs_manager_cfg.n_points
            )
            
        case _: # This is the default (equivalent to 'else')
            raise ValueError(f"Observationmanager {name} not known.")    
    
    if False:
        obs_manager = ObservationTest(obs_mangager=obs_manager,
                                        colorspace=obs_manager_cfg.colorspace,
                                        convert_torch=obs_manager_cfg.convert_torch,
                                        img_width=obs_manager_cfg.img_width, 
                                        img_height=obs_manager_cfg.img_height,
                                        normalize_obs = normalize,
                                        log_directory="logs/observations", log_frequency=1)
        
    #obs_manager.grayscaleimgs_as_uint8(grayscale_imgs_as_uint8)
    return obs_manager
