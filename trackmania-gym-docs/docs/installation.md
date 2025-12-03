# Installation

## Prerequisites

- Python >= 3.10
- Pytorch >= 2.1
    - If you want to train on GPU you also need CUDA, check the official [PyTorch Website](https://pytorch.org/get-started/locally/)
- At least 16GB RAM, the more the better
- [Trackmania Nations Forever](https://store.steampowered.com/app/11020/TrackMania_Nations_Forever/)
- [TrackMania ModLoader](https://tomashu.dev/software/tmloader/), also referred to as TMLoader
    - [TMInterface](https://www.donadigo.com/tminterface/) for communication with the game

## Project Installation

Clone the repository and check that you have fulfilled all Prerequisits. We highly recommend to use a virtual environment like [conda](https://www.anaconda.com/docs/getting-started/miniconda/main).

```sh
git clone https://github.com/dersiwi/trackmania-gym.git && cd trackmania-gym
conda env create -f conda_env.yaml
conda activate tmenv
# now install correct pytorch version (see website), should look something like
# pip install torch torchvision torchaudio --index-url https://download.pytorch...
```

### Platform-CFG

Before starting up the code, create a `platforms.yaml`, containing the operating system, the device and paths to angel-script plugin as well as TM-Forever folder;

```yaml
platforms:
  os: windows / linux                             # operating system
  tmloader: .../TMLoader/TMLoader.exe             # path to tmloader-executable
  plugin: .../TMInterface/Plugins/python_link.as  # path to plugin
  map_dir: .../TmForever/Tracks/Challenges        # path to challenges folder
  device: cuda / cpu                              # device; either cpu or cuda
```
The default location for this file is in `configs/platforms.yaml`, but you can chance this by changing the `platforms_config_path` attribute of `configs/train.yaml` file to the desired path.



### If on Linux

TODO

## Game-Setup

This is optional, but in order for your game not to be on full-screen the whole time, you need to launch the game, as if you'd want to play it and click on `configure` (as of 2025, this button is to the right of the play button). And then set your settings (i.e. winodw-resolution) as you'd like; just unclick fullscreen.
<center>
    <img src="../images/game-configuration.png">
</center>

### Does everything work?
If you have installed everythingh correctly and placed he platform-cfg into the right location, then you should be able to execute the scripts. Try manual-stepping:
```sh
python scripts/tests/manual_stepping.py
```