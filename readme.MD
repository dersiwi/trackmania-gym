
<h1 align="center">Trackmania-Gym</h1>
<p align="center">Trackmania Nations Forever gets a Gymnasium Compatible Environment</p>

<p align="center">
  <a href="https://your-docs-url.com">
    <img src="https://img.shields.io/badge/documentation-blue?style=for-the-badge&logo=readthedocs&logoColor=white" />
  </a>
  <a href="https://discord.gg/yourserver">
    <img src="https://img.shields.io/badge/train_your agent-brightgreen?style=for-the-badge" />
  </a>
</p>


# Make Car Go Fast

In this student research-project by the [ALR](https://alr.iar.kit.edu/) the goal was to investigate on how a reinforcement learning agent can learn in a [TMNF : Trackmania Nations Forever Environment](https://www.trackmaniaforever.com/).

Trackmania-Gym is a Gymnasium Compatible Reinforcement Learning Framework for Trackmania Nations Forever, designed for real-time racing, agent analysis and autonomous driving research.

This framework offers a modular design for experimenting with observations, rewards and terminations, allowing you to focus on actually designing your agent, and not getting lost in annoying implementation-details.

## Provisional nature 

We are still actively working on this project and would appreciate any hints or bugs that we might have missed.

## Experimental Results

Our experiments show that Trackmania-Gym can train agents that reach at least human-like performance.
These agents are also able to generalize to unseen tracks, although with some  reduction in performance.


![Simple-Validated trained agent](images/simple-validated-run.gif){ width="400", .center }

This is one of the agents we traied using Trackmania-Gym, in only around 5 hours on the map simple_validated1. Beware this was not our final agent, much rather an example.
