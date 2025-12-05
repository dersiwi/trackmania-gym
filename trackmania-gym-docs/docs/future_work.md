

## Reference Points and Generalization
One important consideration is how we handle the reference line, especially if we aim to improve generalization in the future. Ideally, for true generalization, the goal would be to train an agent on multiple tracks and then test it on a completely unseen track, expecting it to drive well without additional guidance. However, in our current setup, the agent relies heavily on having a reference line, even on new, unseen tracks. This reliance somewhat contradicts the idea of zero-shot driving, where the agent should generalize without needing such prior information.

A compelling example addressing this issue is presented in [this paper](https://arxiv.org/pdf/2406.12563v1) where the authors trained an actor-critic setup for racing: during training, the critic had access to the reference point, but the actor never received it as input. This design encourages the actor to learn policies that generalize better without being directly dependent on reference lines.

Alternatively, another approach could involve predicting upcoming track points using model predictive control (MPC) or similar techniques, allowing the agent to anticipate the road ahead rather than rely on a predefined path.

## Model Predictive Control 
Model Predictive Control (MPC) appears to be a promising approach for this racing setting here. It allows the agent to plan actions in hindsight by simulating multiple possible future trajectories and then selecting the best one.

Interesting materials:

- https://www.youtube.com/watch?v=19QLyMuQ_BE
- https://www.youtube.com/watch?v=XaD8Lngfkzk

## GRU / LSTM / Transformer

Think aboout whether other, more advanced architectures might be more powerful. I have also heard that just using a very powerful vision backbone, impressive results have been achieved (robot control task tho)