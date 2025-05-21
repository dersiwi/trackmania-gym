import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # TODO : <- i don't want this here and it shouldnt have to be here!!!
from envs.single_agent_env import TMNF_Single_Agent_Env
from game_interaction import game_instance_manager2 as gmi
from pathlib import Path
import os
import gymnasium as gym
from contextlib import redirect_stdout
import numpy as np

if __name__ == "__main__":
    home_dir = os.environ['HOME']
    TMLoader_path = Path(home_dir) / ".wine" / "drive_c" / "Program_Files_x86" / "TmNationsForever" / "TMLoader.exe"
    path_to_plugin = Path(home_dir) / "Documents" / "TMInterface" /  "Plugins" / "Python_Link.as"

    GIM = gmi.GameInstanceManager.get_instance(
        TMLoader_path = TMLoader_path,
        path_to_plugin = path_to_plugin,
        linux = True,
        headless= False)

    # TODO do i really need the port in the environment
    tm_env = TMNF_Single_Agent_Env(
        img_width=100,
        img_height=100,
        port="port",
        observations_space=  gym.spaces.Dict(spaces={"agent": gym.spaces.Box(0, 4, shape=(2,), dtype=int)}),
        gim=GIM,
        map_to_load="ESL-Hockolicious.Challenge.Gbx",
        user_profile=1)
    
    obs: gym.spaces.Dict = tm_env._get_obs();
    with open('out.txt', 'w') as f:
        with redirect_stdout(f):
            for i , (k,v) in enumerate(obs.items()):
                print(k)
                print(v)
                print("-"*20)
    for i in range(100):
        tm_env.step(np.random.randint(0,12))