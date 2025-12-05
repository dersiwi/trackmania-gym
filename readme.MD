
<h1 align="center">Trackmania Gym</h1>
<p align="center">Trackmania Nations Forever finally gets a Gymanasium Compatible Environment</p>

<p align="center">
  <a href="https://your-docs-url.com">
    <img src="https://img.shields.io/badge/documentation-blue?style=for-the-badge&logo=readthedocs&logoColor=white" />
  </a>
  <a href="https://discord.gg/yourserver">
    <img src="https://img.shields.io/badge/train_your agent-brightgreen?style=for-the-badge" />
  </a>
</p>



# Table of Contents
- Communication in game_interaction/readme.MD
- [TMNF as Gymnasium-Environment](#TMNF-as-Gym-Environment)
  - [Environment Structure](#Environment-Structure)

# Setup

Create conda environment using `conda env create -f conda_env.yml`.

## Platform
Before starting up the code, create a `platforms.yaml`, containing the operating system, the device and paths to angel-script plugin as well as TM-Forever folder;

```yaml
platforms:
  os: windows / linux                             # operating system
  tmloader: .../TMLoader/TMLoader.exe             # path to tmloader-executable
  plugin: .../TMInterface/Plugins/python_link.as  # path to plugin
  map_dir: .../TmForever/Tracks/Challenges        # path to challenges folder
  device: cuda / cpu                              # device; either cpu or cuda
```
The default location for this file is in `configs/platforms.yaml`, but you can chance this by changing the `platforms_config_path` attribute of `configs/train.ymal` file to the desired path.

### Reference Line
As of now, reference line are loaded dynamically. Therefore are expected to exist in `trackmania-gym/tracks/referernce_line` and also to be named exactly like the track. I.e. if the track is named Level1.Challenge.Gbx, reference line must be named Level1.npy.

# Make Car Go Fast

In this student research-project at the  [ALR](https://alr.iar.kit.edu/) the goal was to investigate on how a reinforcement learning agent can learn in a [TMNF : Trackmania Nations Forever Environment](https://www.trackmaniaforever.com/).

## TMNF as Gym-Environment

In order to use existing RL-Libraries, like sb3, our first goal was to create a Gymnasium environment for TMNF.  For an in depth tutorial on how to create environment instances see [here](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).

To accomplish this, two seperate processes are initiated; one learner process,
which handles interaction with the environment directly, and one TMInteraction wrapper, which handles interactions with the TMNF-Game-Instance. The communication between learner process
and TMInteraction-Wrapper is done via a specified Inter-Process-Communication protocol. This structure supports modularity by isolating the environment logic from the core game 
communication. Since the interface to the game remains consistent across all environments, the 
communication layer can be reused—allowing you to focus on customizing environment behavior 
without duplicating low-level logic.

For the communication betwen the TMInterface Wrapper and the TMNF-Game instance we use [TMInterface](https://donadigo.com/tminterface/). Furthermore, we also used components of [Linesight](https://github.com/Linesight-RL/linesight/blob/main/trackmania_rl/tmi_interaction/game_instance_manager.py); which components are mentioned in the code.

However, since lineisght has a very difficult codebase, we decided not to continue working on the linesight code but rather salvage the most useful groundwork and move on. We thank you.


### Environment Structure

The environments are defined in `envs`, where `single_agent_env` is the baseclass for other environments. This class handles communication with the ProcessWrapper.

Other modules, like observation_manager and reward_calculation contain interfaces whoose implementation are also in `rewards` or `observations`. The specific implementation is specified in the configuration and the correct implementation of Reward-Calculator or Observation-Manager is given to the environment. This modular approach has been chosen to rapidly switch out Observation, or Reward-Calculation without modifiying the enviornment.

```
trackmania_env
├── envs
│   ├── single_agent_env.py
│   └── testenv_single_agent.py
├── observations
│    └── observation_manager.py
|    └── implementations (contains all implemented observation-managers)
│        ...
├── rewards
│   └── reward_calculation.py
|   └── implementations (contains all implemented reward-managers)
│       ...
└── utils
    ├── position_buffer.py
    └── reference_line_manager.py
        ....
```

`utils` contains standalone classes or methods that are either used by the enviornmenr, reward or observation-managers.
