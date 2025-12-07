# Managers

In order to keep the calculation of Terminations, Rewards and Observations as modular as possible, we created a Term-Manager-Architecture. This means all three, Observation-, Reward-, and Terminationmanager hold terms, which implement the actual calculation. Each manager has a list of Terms and puts together into one unified output. 

All managers extend `trackmania_env.manager.Manager` and all Terms extend `trackmania_env.manager.ManagerTerm`. These baseclasses mostly implement the passing of the environment-variable, as each Manager and Term is given an instance of the environment after instanciation, which it can access using `self.env`. This is very useful for accessing position-buffers (or other) and the ReferenceLineManager of the envrionment.

The calculation-interface is implemented by the specific Term, i.e. `ObservationTerm`. 
 


## Observation-Manager `trackmania_env.observations.observation_manager.ObservationManager`
The observation manager consists of a list of `ObservationTerm`. At the beginning of the step method, the observation is given the raw-observations by the environment, which he then propagates to all its Observation-Terms. These then return the processed observations.

### Observation-Spaces
The observation-manager supports two types of [observation-spaces](https://gymnasium.farama.org/api/spaces/), a Dictionary-Observation-Space and a Box-Observation-Space. In case of the dictionary observation space, the keys of the dictionary are the names of the observation-terms and the valuse are the processed observation by this term. In case of the Box-Observationspace the collection functions similar to the Dictionary-Observation but the Observation-Manager flattens and stacks the Observations into a single Box-Observation.

It is also possible to switch between observation spaces outside of the environment (this may be useful for storing Box-Observation in a replay-buffer, but using Dictionary-Observation in a feature-extractor). To switch the space of the Observation after obtaining it from the environment use:

```py
from trackmania_env.utils.spacetransform import SpaceTransformer
obstransform = SpaceTransformer.get_instance()
# get a box observation, turn it into dictionary and back into box (vectorized case)
boxobs, _, _, _, _ = tm_env.step(action)
obsdict = obstransform.numpy_to_dict_vectorized(o)              # dict_to_numpy(..) (non-vectorized)
backtoboxobx = obstransform.dict_to_numpy_vectorized(obsdict)   # numpy_to_dict(..) (non-vectorized)
```

#### Vectorized Observations

In case of training vectorized, both the Dictionary-Observation-Space and a Box-Observation-Space, are vectorized in the first dimension, i.e. have sahpe `(N, obs_dim)` in case of the Box-Space and `(N, ...)` for each observation-term in case of the Dictionary-Space. `N` is the number of environments.



### Observation-Term `observations.observation_term.ObservationTerm`

Every obsertvation term implements this abstract class, where the most important method are `_get_obs(obs)` and `_normalize(obs)`. The observation-collection
inside the observation-term works like this:
```py
obs, info = self._get_obs(raw_observations, **kwargs)   #this needs to be implemented by every term
if self.normalize:  
    obs = self._normalize(obs)                          #this needs to be implemented by every term
return obs, info
```
Other methods in the abstract class like `flatten()`, `get_flatten_dim()`, `get_native_shape()` exist in order to flatten and potentially rebuild the observation-term, like so:
```py
flattened_obs = obsterm.flatten(native_obs)
assert flattened_obs.shape[0] == obsterm.get_flatten_dim()
native_obs = flattened_obs.reshape(obsterm.get_native_shape())
```
This is necessary in order to switch between Dictionary- and Box-Observation-Space. Feel free to take a look at one of the implemented ObservationManagers in `observations.implementations` on how a Observation-Manager is instanciated.

#### Special Observationterms

- `VectorlikeTerm` : If the native shape of an Observation-Term is `(N,)`, then it can extend this class. This class already implements `flatten()`, `get_flatten_dim()`, `get_native_shape()`.
- `GroupedObservationTerm` : This is a grouping term for `VectorlikeTerm`, which is mostly used in Instanciation of the Observation-Manger. It is especially useful if you want to process observations in the same Encoder, like e.g. all car-related metadata.


## Reward-Manager `trackmania_env.rewards.reward_calculation.RewardCalculator`
The reward manager basically sums the rewards of all of its term together. The weights (as of course the sum is a weighed sum of rewards), are applied by the term individually.
```py
rew = 0
for term in self.reward_terms:
    termvalue = term.calculate_reward_term(raw_obs, processed_obs, rf, ot)
    rew += termvalue
return rew
```
`rf` == race finished (boolean), `ot` == other terimations (dictionary)

### Reward-Term `rewards.reward_calculation.RewardTerm`

Every reward-term needs to implement `_get_term(raw_obs, processed_obs, rf, ot)` which calculates the reward using its input, the environment or nothing (i.e. static 'alive'-rewards). This calculation is then used by the RewardTerm-class to be clipped and weighted:

```py
termval = self._get_term(observations, processed_obs, rf, ot)
clipped_val = np.clip(termval, a_min = self.clip_min, a_max=self.clip_max)
return clipped_val * self.weight
```

### Reward-Normalization `rewards.normalizer.RewardNormalizer`
If you set normalization of reward to true, this noramlizes the rewards using running average and variance 

\[
\frac{r - \mu}{\sqrt{\sigma} + \varepsilon}.
\]

Update Running average and variance after normalizing.

## Termination-Manager `trackmania_env.terminations.termination_maanger.TerminationManager`
The termination manager also collects its reward terms and builds two final outputs; `terminated` and `truncated` each term returns a tuple of both, specifying if any is true. One speciality about the termination manager is that it has two default terms:

- `stuck`   : Terminates the episode if the car is stuck for a certain amount of environment-step (stuck is defined by not moving for more than a specified amount of abosolute distance, can be changed in config. This is actually calculated in the position buffer of the environment by `moved_more_than_threshold`.).
- `timeout` : Maximal number of environment step before the episode terminates (necessary in every MDP, if you seriously want to avoid it, set it to a billion).