
# Environment

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


### Vectorized Environment

We also implemented a vectorized version of the environment, which allows you to collect rollouts from multiple game-instances at the same time, speeding up training time. Training with a vectorized environment and stable-baselines3 is implemented in 
```sh
python scripts/sb3_train_vectorized.py
```
A script for testing and manual inputs to the environment is implemented in 
```
python scripts/tests/step_vectorized.py
```

#### Spaces
The vectorized environment expects action in the shape of `(N,)`, returns rewards and terminations in the same shape, if `N` is the number of environments. The info returned by the vecroized is a list with `N` entries, one for each environment. The shape of vectorized observations are documented with the [observataion-manager](managers.md#vectorized-observations).

#### Resources
For each environment, a new game-instances is spawned. Therefore you have to test how many environments your system can handle. On a system using 32 GB of Ram and a 5070 TI (16GB VRAM) 5 environments seem to work fine.

#### Multiple tracks
The vectorized environment also supports the stepping on multiple tracks, i.e. you can pass 10 track but only spawn 5 environmetns, in order to diversify your domain.


## Continuous Actions 
We have also begun implementing a continous actionspace. As of now; this, and can be used via `scripts/tests/step_continuous.py`. 
!!! Note
    This is still in early development. Once we confirm it works as intended workings will be described here. Until then, code is your friend.

### 2D Continuous

In this mode, the action-space is two dimensional, i.e. $a \in [-1,1]^2$. The first index is the steering, the second one the acceleration. For steering, less than zero is left, >= 0 is right. For acceleration; greater than 0 is gas, smaller than 0 is breaking.

The specifics of the continuization will be documented after more tesing.


### 4D Continuous

Not implemented yet.