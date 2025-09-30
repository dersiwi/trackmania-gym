from trackmania_env.observations.observation_manager import ObservationManager
from trackmania_env.observations.observation_test import ObservationTest
 
from configs.config import TrainConfig, ObservationManagerConfig

from trackmania_env.observations.implementations.linesight_obs_wrapper import get_linesight_obs_instance
from trackmania_env.observations.implementations.nextpoint_obs import NextPointObsManager
from trackmania_env.observations.implementations.sophy_obs import SophyObsManager
from trackmania_env.observations.implementations.dyn_nextpoint_obs import DynamicNextPointObsManager

def get_observation_manager(cfg : TrainConfig, wrap_obs_in_test : bool = False, normalize : bool = False, grayscale_imgs_as_uint8 : bool = False) -> ObservationManager:
    """Instanciate Configuration manager according to confguration.
        - wrap_obs_in_test : If True, Instanciates ObservationTest() with given instanciated observation wrapper
        - grayscale_imgs_as_uint8 : If True and images are grayscale, observation wrapper returns [0,255] value images as uint8, instead of [0,1] as float."""
    
    obs_manager_cfg : ObservationManagerConfig = cfg.rl_env.obs_manager
    name = obs_manager_cfg.name 
    match name:
        case "linesight":
            obs_manager = get_linesight_obs_instance(cfg)

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
                                        normalize_obs = normalize,
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