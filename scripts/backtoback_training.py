"""
Given a bunch of configurations, run back to back training-scripts, without human supervision.
"""

import os

folder_path = r"C:\Users\siwis\Documents\makecargofast\trackmania-gym\configs\trainings_configs"
yaml_files = [f for f in os.listdir(folder_path) if f.endswith(".yaml")]

for config in yaml_files:
    os.system(f"python scripts/train.py --config-path={folder_path} --config-name={config}")
    #TODO : Make sure game is actually closed.