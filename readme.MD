
<h1 align="center">Trackmania-Gym</h1>
<p align="center">Trackmania Nations Forever gets a Gymnasium Compatible Environment</p>

<p align="center">
  <a href="https://dersiwi.github.io/trackmania-gym/">
    <img src="https://img.shields.io/badge/documentation-blue?style=for-the-badge&logo=readthedocs&logoColor=white" />
  </a>
  <a href="https://dersiwi.github.io/trackmania-gym/build-your-agent/">
    <img src="https://img.shields.io/badge/train_your agent-brightgreen?style=for-the-badge" />
  </a>
</p>


# Make Car Go Fast

In this student research-project by the [ALR](https://alr.iar.kit.edu/) the goal was to investigate on how a reinforcement learning agent can learn in a [TMNF : Trackmania Nations Forever Environment](https://www.trackmaniaforever.com/).

Trackmania-Gym is a Gymnasium Compatible Reinforcement Learning Framework for Trackmania Nations Forever, designed for real-time racing, agent analysis and autonomous driving research.

We provide our research reports, [the first report](Make_Car_Go_Fast_Lab_Report.pdf) focussing on validation of the framework and general applicability of on- and off-policy algorithms,  

<p align="center">
    <a href="Make_Car_Go_Fast_Lab_Report.pdf">
        <img src="trackmania-gym-docs/docs/images/reports/Make_Car_Go_Fast_Lab_Report_unroll.png", width="400" class="center">
    </a>
</p>

while [the second one](Make_Car_Go_Fast_Lab_Report.pdf) explores novel algorithmic ideas, domain randomization and vectorized training on multiple tracks
<p align="center">
    <a href="Make_Car_Go_Fast_Lab_Report2.pdf">
        <img src="trackmania-gym-docs/docs/images/reports/Make_Car_Go_Fast_Lab_Report2_unroll.png", width="400" class="center">
    </a>
</p>


This framework offers a [modular design](https://dersiwi.github.io/trackmania-gym/project-structure/) for experimenting with observations, rewards and terminations, allowing you to focus on actually designing your agent, and not getting lost in annoying implementation-details.

## Advancements

This framework offers vectorized training, i.e. Rollout-Collection on multiple environments (tracks) at the same time and continuization of actions over time, allowing RL algorithms like SAC.

We are still actively working on this project and would appreciate any hints or bugs that we might have missed.

Thank you for your interest and have fun.

## Vision based results 

Our experiments show that Trackmania-Gym can train agents that reach at least human-like performance; our agents were furthermore able to generalize to unseen tracks.


<p align="center">
    <img src="trackmania-gym-docs/docs/images/simple-validated-run.gif", width="400" class="center">
</p>

This is one of the agents (End-to-End Vision->Control Policy, Continuous Action-Space) we trained using an using Trackmania-Gym, in only around 5 hours on the map simple_validated1.

## Qucikstart

This is a uv-managed repository; `uv sync` is mandatory. To train, run `uv run sb3train`.

Follow the [setup-instructions](trackmania-gym-docs/docs/installation.md) for setting up the game and platform file correctly.

## License
This project is licensed under the GNU General Public License v3.0.
See the LICENSE file for details.

Copyright (C) 2025 Simon Wilhelm Schübel <br>
Copyright (C) 2025 Tahmur Hassan Mahmood
