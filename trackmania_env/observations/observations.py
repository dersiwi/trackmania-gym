from trackmania_env.observations.observation_manager import ObservationManager
from trackmania_env.observations.listbased_state_obs import ListbasedObs
from trackmania_env.observations.observation_test import ObservationTest
 
from configs.config import TrainConfig, ObservationManagerConfig

from trackmania_env.observations.linesight_obs_wrapper import get_linesight_obs_instance
from trackmania_env.observations.nextpoint_obs import NextPointObsManager
from trackmania_env.observations.sophy_obs import SophyObsManager

def get_observation_manager(cfg : TrainConfig, wrap_obs_in_test : bool = False) -> ObservationManager:
    """Instanciate Configuration manager according to confguration.
        - wrap_obs_in_test : If True, Instanciates ObservationTest() with given instanciated observation wrapper."""
    
    obs_manager_cfg : ObservationManagerConfig = cfg.rl_env.obs_manager
    name = obs_manager_cfg.name 
    match name:
        case "basic":
            obs_manager = ListbasedObs(observation_list=obs_manager_cfg.observation_list, 
                                        colorspace=obs_manager_cfg.colorspace,
                                        convert_torch=obs_manager_cfg.convert_torch,
                                        img_width=obs_manager_cfg.img_width, 
                                        img_height=obs_manager_cfg.img_height)
        case "linesight":
            obs_manager = get_linesight_obs_instance(cfg)

        case "nextpointobs":
            obs_manager = NextPointObsManager(colorspace=obs_manager_cfg.colorspace,
                                        convert_torch=obs_manager_cfg.convert_torch,
                                        img_width=obs_manager_cfg.img_width, 
                                        img_height=obs_manager_cfg.img_height)
        
        case "sophy": 
            obs_manager = SophyObsManager(colorspace=obs_manager_cfg.colorspace,
                                        convert_torch=obs_manager_cfg.convert_torch,
                                        img_width=obs_manager_cfg.img_width, 
                                        img_height=obs_manager_cfg.img_height,
                                        maxlen_history= obs_manager_cfg.maxlen_history,
                                        lookahead_sec = obs_manager_cfg.lookahead_sec,
                                        n_points = obs_manager_cfg.n_points)
            
        case _: # This is the default (equivalent to 'else')
            raise ValueError(f"Observationmanager {name} not known.")    
    
    if wrap_obs_in_test:
        obs_manager = ObservationTest(obs_mangager=obs_manager,
                                        colorspace=obs_manager_cfg.colorspace,
                                        convert_torch=obs_manager_cfg.convert_torch,
                                        img_width=obs_manager_cfg.img_width, 
                                        img_height=obs_manager_cfg.img_height,
                                        log_directory="logs/observations", log_frequency=1)
    return obs_manager