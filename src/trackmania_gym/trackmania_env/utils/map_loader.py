import numpy as np
from pathlib import Path
from typing import Tuple
from pygbx import Gbx, GbxType
import numpy as np


def load_map_with_extrapolated_centers(
    n_before: int,                # Number of points to extrapolate before start
    n_after: int,                 # Number of points to extrapolate after end
    path_to_file: str,                # Name of the .npy file to load

) -> np.ndarray:
    """
    Loads a centerline from file and extrapolates zone centers before the start and after the end.

    Parameters:
        n_before (int): Number of extra points to add before the start.
        n_after (int): Number of extra points to add after the end.
        filename (str): Name of the .npy file to load.
        map_dir (Path): Directory containing the map file.

    Returns:
        np.ndarray: Extended and smoothed zone center points, shape (N + n_before + n_after, 3)
    """
    # Load original zone centers from file
    centers = np.load(path_to_file)

    # Extrapolate before the start
    direction_start = centers[0] - centers[1]
    extra_before = centers[0] + np.expand_dims(direction_start, axis=0) * np.expand_dims(np.arange(n_before, 0, -1), axis=1)

    # Extrapolate after the end
    direction_end = centers[-1] - centers[-2]
    extra_after = centers[-1] + np.expand_dims(direction_end, axis=0) * np.expand_dims(np.arange(1, n_after + 1), axis=1)

    # Stack all together: [before] + [original] + [after]
    centers = np.vstack((extra_before, centers, extra_after))

    # Smooth the centerline 
    centers[5:-5] = 0.5 * (centers[:-10] + centers[10:])

    return centers


def precalculate_virtual_checkpoints_information(zone_centers) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    zone_centers is a 2D array of shape (n_points, 3), containing a list of points on the centerline of the map.
    During the rollout, we will need to use the middle between two consecutive zone_centers.
    We precalculate the coordinates of these middle positions in the "zone_transitions" array.
    If we are in zone_centers[i]:
    - We will calculate distance advanced on segment (zone_transitions[i-1], zone_transitions[i])
    - distance_between_zone_transitions[i-1] represents the length of the current segment (zone_transitions[i-1], zone_transitions[i])
    - distance_from_start_track_to_prev_zone_transition[i-1] contains the sum of segments until zone_transitions[i-1]
    """
    zone_transitions = 0.5 * (zone_centers[1:] + zone_centers[:-1])  # shape: (n_points - 1, 3)
    delta_zone_transitions = zone_transitions[1:] - zone_transitions[:-1]  # shape: (n_points - 1, 3)
    distance_between_zone_transitions = np.linalg.norm(delta_zone_transitions, axis=1)  # shape: (n_points - 2, )
    distance_from_start_track_to_prev_zone_transition = np.hstack(
        (0, np.cumsum(distance_between_zone_transitions))
    )  # shape: (n_points - 1, )
    normalized_vector_along_track_axis = delta_zone_transitions / np.expand_dims(
        distance_between_zone_transitions, axis=-1
    )  # shape: (n_points - 2, 3)
    return (
        zone_transitions,
        distance_between_zone_transitions,
        distance_from_start_track_to_prev_zone_transition,
        normalized_vector_along_track_axis,
    )

def gbx_to_raw_pos_list(gbx_path: Path):
    """
    Read a .gbx file, extract the raw positions of the best ghost included in that file.
    """
    gbx = Gbx(str(gbx_path))
    ghosts = gbx.get_classes_by_ids([GbxType.CTN_GHOST])
    assert len(ghosts) > 0, "The file does not contain any ghost."
    ghost = min(ghosts, key=lambda g: g.cp_times[-1])
    if ghost.num_respawns != 0:
        print("")
        print("------------    Warning: The ghost contains respawns  ---------------")
        print("")
    records_to_keep = round(ghost.race_time / ghost.sample_period)

    print(ghost.race_time, f"ghost has {len(ghost.records)} records and {len(ghost.control_entries)} control entries")
    print("Keeping", records_to_keep, "out of", len(ghost.records), "records for a race time of", ghost.race_time / 1000)

    raw_positions_list = []
    for r in ghost.records[:records_to_keep]:
        raw_positions_list.append(np.array([r.position.x, r.position.y, r.position.z]))

    return raw_positions_list

def get_checkpoint_positions_from_gbx(map_path: str):
    """
    Given a challenge.gbx file, return an unordered list of the checkpoint positions on that track.
    <!> Warning: this function assumes that the block size for that map is 32x8x32. This is true for campaign maps, but not for all custom maps.
    """
    g = Gbx(map_path)

    challenges = g.get_classes_by_ids([GbxType.CHALLENGE, GbxType.CHALLENGE_OLD])
    if not challenges:
        quit()

    checkpoint_positions = []
    challenge = challenges[0]
    for block in challenge.blocks:
        if "Checkpoint" in block.name or "Line" in block.name:
            checkpoint_positions.append(np.array(block.position.as_array(), dtype="float"))
            if "High" in block.name:  # Added for E03
                checkpoint_positions[-1] += np.array([0, 7 / 8, 0])
            elif block.name in ["StadiumRoadMainCheckpointRight", "StadiumRoadMainCheckpointLeft"]:  # Added for "exceed my tech"
                checkpoint_positions[-1] += np.array([0, 5 / 8, 0])
    checkpoint_positions = np.array(checkpoint_positions) * np.array([32, 8, 32]) + np.array((16, 0, 16))
    return checkpoint_positions


def sync_virtual_and_real_checkpoints(zone_centers: np.ndarray, map_path: str, sync_virtual_and_real_checkpoints = True):
    """
    Given a challenge.gbx file and a list of VCP, return:
        - next_real_checkpoint_positions: a list of points with the same length as the list of VCP
        - max_allowable_distance_to_real_checkpoint: a list of distances with the same length as the list of VCP

    In this function we match each checkpoint with its corresponding closest VCP.
    In game_instance_manager.py, we will enforce that the car can only advance towards the next VCP if it was within 12 meters of the center of the real checkpoint.
    """
    next_real_checkpoint_positions = np.zeros((len(zone_centers), 3))
    max_allowable_distance_to_real_checkpoint = 9999999 * np.ones(len(zone_centers))
    if sync_virtual_and_real_checkpoints:
        checkpoint_positions = get_checkpoint_positions_from_gbx(map_path)
        for checkpoint_position in checkpoint_positions:
            dist_vcp_cp = np.linalg.norm(zone_centers - checkpoint_position, axis=1)
            while np.min(dist_vcp_cp) < 12:
                # This while is necessary for multi-lap maps, to identify the multiple VCP that are linked to the same CP
                idx = dist_vcp_cp.argmin()
                next_real_checkpoint_positions[idx, :] = checkpoint_position
                max_allowable_distance_to_real_checkpoint[idx] = 12
                dist_vcp_cp[max(0, idx - 200) : idx + 200] = 99999

    return next_real_checkpoint_positions, max_allowable_distance_to_real_checkpoint