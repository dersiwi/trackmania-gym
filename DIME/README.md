The Code for the DIME paper submission at ICML2025.
## DIME: Diffusion-Based Maximum Entropy Reinforcement Learning 

This repository accompanies the paper "[DIME: Diffusion-Based Maximum Entropy Reinforcement Learning](https://arxiv.org/pdf/2502.02316)" published at ICML 2025.

### Learning Curves are available in *paper_results* 
**Update:** We have uploaded all the learning curve data to our repository. You can find the data in the *paper_results* folder.
Additionally, we have run DIME on all remaining DMC enviroinments and added the results to the same folder. 

### Installation
The file setup.sh provides a convenient way to set up the conda environment and install the required packages automatically via
```bash
chmod +x setup.sh
./setup.sh
```

After installation is finished, the conda environment can be activated, and the code can be run using 

```python
python run_dime.py
```

### Running DIME

Specific parameters can be set in the terminal such as the learning environment using hydra's multirun function

```python
python run_dime.py --multirun env_name=dm_control/humanoid-run
```

Detailed hyperparameter specifications are available in the config directory.
The current config file is adapted to the hyperparameters used for DMC. If you want to run DIME on the gym environemtns, 
you only need to change the v_min and v_max parameters of the critic as specified in the appendix of the paper. We used the same values for both gym environments. 
For example if you would like to run DIME on gym's Humanoid-v3 environment, you can do so by running

```python
python run_dime.py env_name=Humanoid-v3  alg.critic.v_min=-1600 alg.critic.v_max=1600
```


## Acknowledgements
Portions of the project are adapted from other repositories: 
- https://github.com/DenisBless/UnderdampedDiffusionBridges is licensed under MIT,
- https://github.com/adityab/CrossQ is licensed under MIT and is built upon code from "[Stable Baselines Jax](https://github.com/araffin/sbx/)"