# Welcome to the Documentation of Trackmania-Gym

Trackmania-Gym is a Gymnasium Compatible Reinforcement Learning Framework for Trackmania Nations Forever, designed for real-time racing, agent analysis and autonomous driving research.

This framework offers a modular design for experimenting with observations, rewards and terminations, allowing you to focus on actually designing your agent, and not getting lost in annoying implementation-details.

## How to read

If installation is complete, continue with
1. [Reference Line](refline.md) - outlaying the fundamentals of our measure of progress along the track
2. [The Structure of the project](project-structure.md) - undersand how the framework is organized and get a feel for the environment
3. [The Managers](managers.md) - as the basic building blocks of the environment 
4. [Our testing framework](testing.md) - essential for live visualization of the output of the managers

## Config-Based Launch

Trackmania-Gym uses [Hydra](https://hydra.cc/docs/intro/) to manage the configuration of our project. The base-configuration is `configs/train.yaml`. (Almost) all scripts in `scripts/` automatically laods theis configuration, which includes all sub-configurations (environment, agent, logging, etc).

Get familiar with the config-setup, by skimming through `train.yaml` and following the paths to other configuration-files, such as `rl_env/single_agent_env.yaml`.

### Hydra-Instanciation
In order to not hardcode values into our code, Trackmania-Gym uses Hydra-instanciation to instanciate certain classes, like for example the `ObservationManager`. Every implemented component has its own configuration file, for example each implemented `ObservationManager` variant corresponds to a YAML file that Hydra uses to build the class at runtime.