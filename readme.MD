# Table of Contents
- Communication in game_interaction/readme.MD
- [Linesight workings (train.py)](#linesight-workings-trainpy)
  - [Worker Process(es)](#worker-processes)
- [Gymnasium Environment](#gymnasium-environment)
  - [Folder Structure](#folder-structure)
- [Open questions](#open-questions)




# Linesight workings (train.py)

In the `train.py` file basically three things happen; 

1. initialization of multi process utilities (share memory i think)
2. start of so called 'worker process' 
3. start of 'learner process'

### Worker Process(es)

The worker processes (file is called `collector_process.py`) are mainly responsible for collecting rollouts. For this, the inference network is shared between the learner processes and the worker process:

```py
with shared_network_lock:
    uncompiled_inference_network.load_state_dict(uncompiled_shared_network.state_dict())
```
More specifically, this netowrk is wrapped by an `Inferer` which performs inference of a single state via the inference network, 
the Inferer is also resposible for enforcing exploration poliy.
The rollout is performed by the `game_instance_manager`, collecting states and frames:

```py
rollout_results = {
    "current_zone_idx": [],
    "frames": [],
    "input_w": [],
    "actions": [],
    "action_was_greedy": [],
    "car_gear_and_wheels": [],
    "q_values": [],
    "meters_advanced_along_centerline": [],
    "state_float": [],
    "furthest_zone_idx": 0,
}
```
# Gymnasium Environment

This section explains how **TrackMania Nations Forever** is wrapped in a custom **Gymnasium environment**.  
The Gymnasium API allows us to define a clean, modular reinforcement learning environment that is compatible with multiple RL algorithms and libraries. For an in depth tutorial on how to create environment instances see [here](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).

## Folder Structure
Everything related to the custom Gymnasium environment for
TrackMania can be found in the `trackmania_env` folder. 
The current structure is as follows:
```
trackmania_env
├── envs
│   ├── __init__.py
│   └── single_agent_env.py
├── __init__.py
└── wrappers
    └── __init__.py
```
The `envs` folder contains different environment implementations for TrackMania. 
These can include single-agent environments, multi-agent setups, or other variations.
This structure supports modularity by isolating the environment logic from the core game 
communication. Since the interface to the game remains consistent across all environments, the 
communication layer can be reused—allowing you to focus on customizing environment behavior 
without duplicating low-level logic.

The `wrappers` folder contains custom **wrapper classes**. Wrappers are a convenient way to modify an 
existing environment’s behavior (e.g., altering observations, rewards, or rendering) without modifying 
its core implementation.
Using wrappers helps reduce boilerplate code and improves modularity. Wrappers can also be **chained together**. For more details, refer to the [Gymnasium documentation on wrappers](https://gymnasium.farama.org/api/wrappers/).

$\color{red}\text{TODO: how the gym env works with the
game instance manager and tminterface for all the mehtods}$ 
## Open questions

- How exactly is data obtained from the race?
- How exactly does a rollout work?; I think a race is simulated until timout?
- Why more than one worker process? Is one not sufficient?
- Is one worker process only performing ONE rollout step?
- how are rollouts triggered repeatately? The train loop (learner process) just has a while True: condition, but i don't quite undestand how a worker process is triggered; i.e. knows when to start performing rollouts
- How many processes are there?