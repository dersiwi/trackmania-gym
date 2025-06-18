
from trackmania_env.observations.linesight_obs_wrapper import LinesightObservationWrapper
from configs.config import LinesightObsCfg, ObservationManager, ImageConfig, RLEnvConfig, TrainConfig

from trackmania_env.utils.map_loader import (
    load_map_with_extrapolated_centers,
    precalculate_virtual_checkpoints_information,
    sync_virtual_and_real_checkpoints)



def get_linesight_obs_instance(cfg : TrainConfig):
    linesightcfg : LinesightObsCfg = cfg.rl_env.linesightobsmanager
    obsconfig : ObservationManager = cfg.rl_env.obsmanager
    image_cfg : ImageConfig = cfg.image
    zone_centers = load_map_with_extrapolated_centers(
        n_before = linesightcfg.n_zone_centers_extrapolate_before_start_of_map,
        n_after  = linesightcfg.n_zone_centers_extrapolate_after_end_of_map,
        path_to_file = cfg.gmi.reference_line,)

    (
        zone_transitions,
        distance_between_zone_transitions,
        distance_from_start_track_to_prev_zone_transition,
        normalized_vector_along_track_axis,
    ) = precalculate_virtual_checkpoints_information(zone_centers)

    (
        next_real_checkpoint_positions,
        max_allowable_distance_to_real_checkpoint,
    ) = sync_virtual_and_real_checkpoints(
        zone_centers, 
        cfg.platforms.map_dir+'/'+cfg.gmi.track,
        linesightcfg.sync_virtual_and_real_checkpoints)


    return LinesightObservationWrapper(observation_list=obsconfig.observation_list, 
                                    colorspace=obsconfig.colorspace,
                                    convert_torch=obsconfig.convert_torch,
                                    img_width=image_cfg.width, 
                                    img_height=image_cfg.height,

                                    zone_centers = zone_centers,
                                    zone_transitions = zone_transitions,
                                    distance_between_zone_transitions = distance_between_zone_transitions,
                                    distance_from_start_track_to_prev_zone_transition = distance_from_start_track_to_prev_zone_transition,
                                    normalized_vector_along_track_axis = normalized_vector_along_track_axis,
                                    next_real_checkpoint_positions = next_real_checkpoint_positions,
                                    max_allowable_distance_to_real_checkpoint = max_allowable_distance_to_real_checkpoint,

                                    cfg = cfg.rl_env.linesightobsmanager)