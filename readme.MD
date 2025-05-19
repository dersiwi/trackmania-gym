# Table of Contents
- [Setup and Communication](#setup-and-communication)
  - [GameInstanceManager](#gameinstancemanager)
  - [Create Python Link](#create-python-link)
    - [Where to get this information?](#where-to-get-this-information)
- [Communication between game (server) and python (client)](#communication-between-game-server-and-python-client)
  - [Sending inputs to control car](#sending-inputs-to-control-car)
  - [Receiving Frames](#receiving-frames)
  - [Getting Simulation state](#getting-simulation-state)
- [Linesight workings (train.py)](#linesight-workings-trainpy)
  - [Worker Process(es)](#worker-processes)
- [Gymnasium Environment](#gymnasium-environment)
  - [Folder Structure](#folder-structure)
- [Open questions](#open-questions)


# Setup and Communication

Downloaded TMNF, the modloader and installed TMInterface as wanted and described by donadigo [here](https://donadigo.com/tminterface/installation), 
more specifically; since this builds upon the [linesight repository](https://github.com/Linesight-RL/linesight) it might make more sense to follow it's [installation guide](https://linesight-rl.github.io/linesight/build/html/installation.html).


### GameInstanceManager

For now, this class is responsible for launching and closing the game via PowerShell commands. (It launches the TMLoader).

### Create Python Link
Now how to run python scripts using this setup?
From what i undestand TMInterface has to launch a TCP-Socket, to which a python instance can connect. 
The [file](https://gist.github.com/Agade09/f5b57048895b0a3a64fb0b1ae09cbd7f) `plugin/python_link.as`  has to be placed into the plugin-folder in the TMInterface folder; `C:\Users\[username]\OneDrive\Dokumente\TMInterface\Plugins`.
I modified the file such that the port is 8775 and not 0; i had to set PORT=8775 in all mentions and remove `PORT = uint16(GetVariableDouble("custom_port"));`, as this did not work for some reason. I also established a connection with a python script, however it timed out every time. 

[IMPORTANT: When launching the linesight respository, it uses `custom_port` inside the angelscript to set the port via the python script. This may be important if multiple instances of the game are running and threrefore need different ports.]

#### Where to get this information?

I went into the discord and it was a pinned answer in the programming channel; this was the exact message:

```
This is a gist with a few files from our private codebase.
https://gist.github.com/Agade09/f5b57048895b0a3a64fb0b1ae09cbd7f
With this, we reproduce a single-thread TMI1. 4.3-like API in Python.
The file python_link.as must be placed in the TMInterface folder.
The file example_script.py is a minimal functioning example showing how to open a connection.
```

## Communication between game (server) and python (client)

They communicate via binary protocol. The first message of the server is always the message type, the second is the payload of the message. 

Theoretically, the trackmania instance is acting as a simulation, sending `SC_RUN_STEP_SYNC`-message everytime a simulation step is completed. 

An example interaction is implemented in `example.py`.


### Sending inputs to control car

The car is controlled not via commands specified in the [TMInteraction mod-documentation](https://donadigo.com/tminterface/) (here, there are commands like `gas 50000` - they also work in the console inside the game when game is launched via modloader). [DISCLAIMER] i did not further explore controlling the car using such commands, since the input states work really well: 

In linesight (i.e. in `tminterface2.py`) controlling the car is done using `set_input_state`, which takes a boolean for left, right, accelerate and brake, basically acting as if a key was pressed on the keyboard. This then sets the value of the control for the next simulation step.

### Receiving Frames

Actually quite straight-forward using tminterface2:

```py
if msgtype == int(MessageType.SC_RUN_STEP_SYNC): # simulation step is complete

    _time = iface._read_int32()

    iface.request_frame(IMG_W, IMG_H) #<--- imgs scaled automatically by trackmania
    
    # do other stuff like send inputs or commands here

    iface._respond_to_call(msgtype)

elif msgtype == int(MessageType.SC_REQUESTED_FRAME_SYNC): # collect frame

    frame : np.npdarray = iface.get_frame(IMG_W, IMG_H)
    iface._respond_to_call(msgtype)
```

### Getting Simulation state

The simulation state queried from the game using `iface.get_simulation_state()` which returns a `SimStateData` object. The example-interaction queries simulation state and prints it to the console. This output looks like this: 

```
=== Simulation State Snapshot ===
    Time:                786130 ms
    Position:            [529.6427612304688, 49.354209899902344, 652.0673828125]
    Velocity:            [50.89628219604492, -0.166009321808815, -0.7100085020065308]
    Rotation (YawPitchRoll): [1.582823502445449, -0.0033017142612521374, -0.004220573944300383]
    Display Speed:       183 units

Input State:
    Accelerate:          True
    Brake:               False
    Left:                False
    Right:               False
    Steer (analog):      0
    Gas (analog):        0

Race Progress:
    Race Time:           783520 ms
    Rewind Time:         783530 ms
    Num Respawns:        0
   Checkpoint Times:    [10850, 773920, -1]
    Checkpoints Passed:  3
```

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