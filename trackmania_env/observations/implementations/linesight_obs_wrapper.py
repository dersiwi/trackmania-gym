"""
This module includes code adapted and refactored from the Linesight-AI project
(https://github.com/Linesight-RL/linesight). Credit to the original authors
for the foundational implementation; refactloring changes made here.
"""
from trackmania_env.observations.observation_manager import DictObservationManager
from trackmania_env.observations.observation_terms.linesight_terms import LinesightDynamicTerm,ZoneCenterFeatures
from trackmania_env.observations.observation_terms.img_terms import ImageObservationTerm
from trackmania_env.observations.observation_terms.basic_terms import MobileStatesTerm
from trackmania_env.observations.observation_term import GroupedObservationTerm

class LinesightObservationManager(DictObservationManager):
    def __init__(
        self,
        colorspace: str,
        img_width,
        img_height,
        ref_line_path,
        map_path,
        convert_torch: bool = True,
        normalize: bool = False,
        n_zone_centers_extrapolate_after_end_of_map: int = 1000,
        distance_between_checkpoints: float = 0.5,
        road_width: int = 90,
        n_zone_centers_in_inputs: int = 40,
        margin_to_announce_finish_meters: int = 700,
        one_every_n_zone_centers_in_inputs: int = 20,
        n_zone_centers_extrapolate_before_start_of_map: int = 20,
        n_prev_actions_in_inputs: int = 5,
        sync_virtual_and_real_checkpoints: bool = True,
    ):
        """
        Initializes the Linesight Observation Manager.

        Parameters:
        -----------
            - colorspace (str, default="grayscale")                           : Colorspace used for rendering images (e.g., "grayscale", "rgb").
            - convert_torch (bool, default=True)                              : Whether to convert the observations (e.g. images) to PyTorch tensors.
            - img_width (int, default=128)                                    : Width of the input images in pixels.
            - img_height (int, default=128)                                   : Height of the input images in pixels.
            - n_zone_centers_extrapolate_after_end_of_map (int, default=1000) : Number of synthetic zone centers to extrapolate beyond the end of the actual map.
            - distance_between_checkpoints (float, default=0.5)               : Distance in meters between consecutive virtual checkpoints.
            - road_width (int, default=90)                                    : Assumed road width in meters; includes margin for safety on curves.
            - n_zone_centers_in_inputs (int, default=40)                      : Number of zone centers to include in the observation input.
            - margin_to_announce_finish_meters (int, default=700)             : Distance margin before the final checkpoint to begin announcing track completion.
            - one_every_n_zone_centers_in_inputs (int, default=20)            : Sampling interval; use one zone center every N centers in the input.
            - n_zone_centers_extrapolate_before_start_of_map (int, default=20): Number of synthetic zone centers to extrapolate before the start of the map.
            - n_prev_actions_in_inputs (int, default=5)                       : Number of previous agent actions (e.g., steering, throttle) to include in the observation input for temporal context.
            - sync_virtual_and_real_checkpoints (bool, default=True)          : Whether to align virtual checkpoints with real-world checkpoints for consistent navigation and evaluation.
        """
        zone_center_features = ZoneCenterFeatures(
            n_extrapolate_before = n_zone_centers_extrapolate_before_start_of_map,
            n_extrapolate_after  = n_zone_centers_extrapolate_after_end_of_map,
            refline_path         = ref_line_path,
            map_path             = map_path,
            sync_virt_and_real_cp = sync_virtual_and_real_checkpoints,
            dist_between_vcp     = distance_between_checkpoints,
            road_width           = road_width,
            zone_input_count     = n_zone_centers_in_inputs,
            zone_stride          = one_every_n_zone_centers_in_inputs,
        )

        super().__init__(
            convert_torch = convert_torch,
            normalize     = normalize,
            observation_terms = [
                ImageObservationTerm(
                    colorspace = colorspace,
                    img_height = img_height,
                    img_width  = img_width,
                ),
                GroupedObservationTerm(name= "floats", observation_terms= [
                    MobileStatesTerm(),
                    LinesightDynamicTerm(),
                    zone_center_features
                ])
            ]
        )