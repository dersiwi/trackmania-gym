# Build your own agent
On this page we quickly describe the steps we take before training our agents

## Checklist

- Decide on Observation-Space and create an `ObservationManager`, if the already implemented ones do not suffice
- Create Reward Function and implement a `RewardManager`, if nececssary
- Check if config is as you'd like, most important configs to check are `configs/train.yaml` and `configs/rl_env/single_agent_env.yaml`. Most importantly, set the managers in the `single_agent_env.yaml`:
```
defaults:
  - obs_manager: next_point_obs.yaml
  - reward_manager: next_point_rewards.yaml
  - termination_manager: timeout.yaml
```

!!! note
    Be sure to check the config-files in detail in order not to miss any important setting; this may really result in unnecessary troubleshooting.

### Check your reward function
We highly suggest you check [your reward function](testing.md#testing-the-reward-function), using the [testing-framework](testing.md) to avoid unncessary errors, or scaling issues, as errors in the reward functions are the most likely reason for lack of learning.

## What to expect
When using a single environment and images in your observation-space, on a reasonably capable GPU, it tatkes around 5-10 hours to see results.