import hydra.utils
from trackmania_env.observations.observation_manager import ObservationManager
from trackmania_env.observations.observation_test import ObservationTest
from configs.config import TrainConfig

def get_observation_manager_from_cfg(cfg : TrainConfig, 
                            wrap_obs_in_test : bool = False, 
                            normalize : bool = False, 
                            grayscale_imgs_as_uint8 : bool = False
                           ) -> ObservationManager:
    """Instantiate ObservationManager directly from config using Hydra."""
    
    obs_manager = hydra.utils.instantiate(
        cfg.rl_env.obs_manager,
        normalize=normalize 
    )

    if wrap_obs_in_test:
        obs_manager = ObservationTest(
            obs_mangager=obs_manager,
            colorspace=cfg.rl_env.obs_manager.colorspace,
            convert_torch=cfg.rl_env.obs_manager.convert_torch,
            img_width=cfg.rl_env.obs_manager.img_width,
            img_height=cfg.rl_env.obs_manager.img_height,
            normalize_obs=normalize,
            log_directory="logs/observations", 
            log_frequency=1
        )
        
    # if hasattr(obs_manager, 'grayscaleimgs_as_uint8'):
    #     obs_manager.grayscaleimgs_as_uint8(grayscale_imgs_as_uint8)

    return obs_manager
