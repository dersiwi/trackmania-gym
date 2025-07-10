from trackmania_env.observations.observation_manager import ObservationManager
from trackmania_env.observations.observation_test import ObservationTest
 
from configs.config import TrainConfig

from trackmania_env.observations.linesight_obs_wrapper import get_linesight_obs_instance
from trackmania_env.observations.nextpoint_obs import NextPointObsManager

def get_observation_manager(cfg : TrainConfig) -> ObservationManager:
    """Instanciate Configuration manager according to confguration."""
    obs_manager_cfg = cfg.rl_env.obsmanager
    if cfg.rl_env.env.obs_manager =="basic":
        obs_manager = ObservationManager(observation_list=obs_manager_cfg.observation_list, 
                                        colorspace=obs_manager_cfg.colorspace,
                                        convert_torch=obs_manager_cfg.convert_torch,
                                        img_width=cfg.image.width, 
                                        img_height=cfg.image.height)
        
    elif cfg.rl_env.env.obs_manager =="linesight":
        obs_manager = get_linesight_obs_instance(cfg)

    elif cfg.rl_env.env.obs_manager == "nextpointobs":
        obs_manager = NextPointObsManager(observation_list=obs_manager_cfg.observation_list, 
                                        colorspace=obs_manager_cfg.colorspace,
                                        convert_torch=obs_manager_cfg.convert_torch,
                                        img_width=cfg.image.width, 
                                        img_height=cfg.image.height)
    else:
        raise ValueError(f"Observationmanager {cfg.rl_env.env.obs_manager} not known.")    
    
    if cfg.rl_env.env.wrap_obs_in_test:
        obs_manager = ObservationTest(obs_mangager=obs_manager,
                                      observation_list=obs_manager_cfg.observation_list, 
                                        colorspace=obs_manager_cfg.colorspace,
                                        convert_torch=obs_manager_cfg.convert_torch,
                                        img_width=cfg.image.width, 
                                        img_height=cfg.image.height,
                                        log_directory="logs/observations", log_frequency=1)
    return obs_manager