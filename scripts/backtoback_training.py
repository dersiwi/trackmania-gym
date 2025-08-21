"""
Given a bunch of configurations, run back to back training-scripts, without human supervision.
"""

import os

folder_path = os.path.join(os.getcwd(), r"configs\trainings_configs")
assert os.path.exists(folder_path)
yaml_files = []
with open("configs/trainings_configs/listfile.txt", "r") as file:
    for i, line in enumerate(file):
        yaml_files.append(line.strip().strip("\n"))
        assert os.path.exists(os.path.join(folder_path, yaml_files[i])), f"Could not find file : '{yaml_files[i]}' "
input(f"Confiform blocktraining on {yaml_files} \n[Press anything to continue]")


for config in yaml_files:
    os.system(f"python scripts/sb3_train.py --config-path={folder_path} --config-name={config}")
    #TODO : Make sure game is actually closed.