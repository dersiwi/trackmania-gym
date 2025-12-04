# Managers

All managers basically function by employing reward-terms. Each term holds an instance of the environment. And supplies the Manager with whatever calculation it did. The manager then combines them accordingly.

## Observation-Manager `trackmania_env.observations.observation_manager.ObservationManager`
The observation manager consists of a list of `ObservationTerm`. At the beginning of the step method, the observation is given the raw-observations by the environment, which he then propagates to all its Observation-Terms. These then return the processed observations.

### Observation-Spaces
The observation-manager supports two types of observation-spaces, a Dictionary Observation space and a Box-Observation-Space. In case of the dictionary observation space, the keys of the dictionary are the names of the observation-terms and the valuse are the processed observation by this term. In case of the Box-Observationspace the collection functions similar to the Dictionary-Observation but the Observation-Manager flattens and stacks the Observations into a single Box-Observation.



### Observation-Term `observations.observation_term.ObservationTerm`

Every obsertvation term implements this abstract class, where the most important method are `_get_obs(obs)` and `normalize()`. The observation-collection
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

Feel free to take a look at one of the implemented ObservationManagers in `observations.implementations` on how a Observation-Manager is instanciated.

#### Special Observationterms

- `VectorlikeTerm` : If the native shape of an Observation-Term is `(N,)`, then it can extend this class. This class already implements `flatten()`, `get_flatten_dim()`, `get_native_shape()`.
- `GroupedObservationTerm` : This is a grouping term for `VectorlikeTerm`, which is mostly used in Instanciation of the Observation-Manger. It is especially useful if you want to process observations in the same Encoder, like e.g. all car-related metadata.


## Reward-Manager `trackmania_env.rewards.reward_calculation.RewardCalculator`
The reward manager basically sums the rewards of all of its term together. Even the weights (as of course the sum is a weighed sum of rewards), are applied by the term individually.
```py
rew = 0
for term in self.reward_terms:
    termvalue = term.calculate_reward_term(raw_obs, processed_obs, rf, ot)
    rew += termvalue
return rew
```
`rf` == race finished (boolean), `ot` == other terimations (dictionary)

### Reward-Term `rewards.reward_calculation.RewardTerm`