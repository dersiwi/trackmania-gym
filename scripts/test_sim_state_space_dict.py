# utils
from contextlib import redirect_stdout
import sys, os
# TODO : <- i don't want this here and it shouldnt have to be here!!!
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

# Hydra related imports
import hydra
import traceback

# imports for communication between TMInterface and environment
from game_interaction.run_multiprocess_wrapper import start_process_and_wait_for_startsignal
from game_interaction.process_wrapper import TMIProcessWrapper

from configs.config import TrainConfig
import numpy as np

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "../configs",
    "config_name": "train.yaml",
}

"""
in this test we initialize the observation space of the environment with all the entries of the 
simstate_space_dict.py and try to do num_steps simulations steps. If everything works no errors should be thrown.
For clarity we dont set the observations in here since per default the observations space of single_agent_env_2 is 
the sim_state_space_dict.
"""
@hydra.main(**_HYDRA_PARAMS)
def main(cfg : TrainConfig):
    num_steps = 20
    # Instanciate GMI, TMNF-Environment and start TMi-Interaction process.
    tmi_process, control_queue, response_queue = start_process_and_wait_for_startsignal(cfg.platforms, cfg.gmi, cfg.image.width, cfg.image.height)

    try:
        tm_env = hydra.utils.instantiate(cfg.rl_env.env)(
            command_queue=control_queue,
            response_queue=response_queue)
        
        with open('test_sim_state_space_dict_out.txt', 'w') as f:
            with redirect_stdout(f):
                for _ in range(num_steps):
                    action_idx = np.random.randint(low=0,high=12)
                    obs,_,_,_,_=  tm_env.step(action_idx)
                    for _ , (k,v) in enumerate(obs.items()):
                        print(k)
                        print(v)
                        print("-"*20)
                    
    except Exception as e:
        traceback.print_exc()

    except KeyboardInterrupt as kinterrupt:
        print("KeyboardInterrupt")

    finally:
        # Finalize training and close game all processes.
        control_queue.put(TMIProcessWrapper.IPCCommands.get_end_syncloop_command(1000)) #1000 doesnt matter.
        tmi_process.join()


if __name__ == "__main__": 
    main()