helpstr = """
This script reads a .gbx file, and creates a list of Virtual CheckPoints (VCP) based on the best ghost found in that file.

The .gbx file may either be a .challenge.gbx, or a .replay.gbx.

The VCP file is saved in base_dir/maps/map.npy. It should typically be renamed manually after running this script.

The distance between virtual checkpoints is currently 50cm (hardcoded).
"""

import argparse
from pathlib import Path

from resources.tracks.reference_line.create_refline.geometry import extract_cp_distance_interval
from trackmania_gym.trackmania_env.utils.map_loader import gbx_to_raw_pos_list

def main():
    parser = argparse.ArgumentParser(description = helpstr)
    parser.add_argument("gbx_path", type=Path)
    parser.add_argument("map_name",type=str)
    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parents[3]

    raw_positions_list = gbx_to_raw_pos_list(args.gbx_path)
    _ = extract_cp_distance_interval(raw_positions_list, 0.5, base_dir,args.map_name)


if __name__ == "__main__":
    main()
