# Project Structure

The two main things to understand when using this project is how the game-interaction works, and how the environment is structured. A high-level overview of how all components work together is given in this image:

<center>
    <img src="../images/structure.png">
</center>

The learner process consists of a learner (most likely a RL-Library like sb3) and the environment. The learner is what steps the environment, i.e. calls the `step`-method. The environment then sends IPC (Inter-Process-Communication) commands to the TMInteraction Process. This process continuously communicates with the plugin provided by TMInteraction via a TCP-Connection.

This architecture was chosen in order to completely de-couple the environment from the necessary continuous communication with the game. This enables the environment to only send and receive on-demand.

## Environment

The environment is implemented as a Gymnasium-Envioronment. For an in depth tutorial on how to create environment instances see [here](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/). 

We chose a task-based manager architecture, where we employ managers to calculate Rewards, Observations and Terminations. Under the hood, the main task of the environment is to send actions to the TMInteraction-Process and gather raw-observations (images and game-states). Each manager, i.e. Reward-, Observation- and Termination-Manager has access to these raw-observations.

In this project, multiple versions of an environment are implemented. The main one that is talked about in this article is `trackmania_env.envs.single_agent_env2.TMNF_Single_Agent_Env`. Other environments that are implemented use this environment. Other environments include:

 - `envs.sec_env.CrashProofEnvironment` this is just a wrapper for the original environment, as we sometimes had problems with the game not resetting after the agent reached the finish-line. This led to crashes during trainings. This wrapper-environment catches these crashes and restarts the whole pipeline. We mainly use this environment.
 - `vectorized.VectorizedTMEnvironment` is a vectorized version of the environment. It stores the environments and exposes a batched interface.
 - `vectorized.SB3Vectorized` is a wrapper-environment for `VectorizedTMEnvironment` that is compatilble with stable-baselines3. This is used in `scripts/sb3_train_vectoried`.



### Order of Operations
The order of operations in the environment in the step-method.

1. Send `step`-command to the Process-Wrapper
2. Advance the reference line
3. Calculate Terminations
4. Calculate Observations
5. Calculate Rewards

### Non-Gym-Interfaces
The environment also 

## Process-Wrapper

