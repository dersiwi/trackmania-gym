import sys, os
# TODO : <- i don't want this here and it shouldnt have to be here!!!
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 

from neural_networks.bro_replay_buffer import VecReplayBuffer
from bro_torch import BRO

import torch
import numpy as np
import random

import wandb

from absl import app, flags

# Hydra related imports
import hydra
import traceback

from hydra.core.hydra_config import HydraConfig
from typing import Optional


from configs.config import TrainConfig

from trackmania_env.envs.vectorized import SB3Vectorized
from trackmania_env.envs.sec_env import CrashProofEnvironment

from utils.hydra_wandb_utils import load_and_merge_platform, secure_attribute_retrieval
from utils.introscreen import introscreen
from utils.experiment_managers.sb3_exp_manager import Sb3ExperimentManager

from tmn_sb3.utils.from_cfg import get_model_from_config
from multiprocessing import Lock

#flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 5, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('eval_interval', 25000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 128, 'Mini batch size.')
flags.DEFINE_integer('max_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('replay_buffer_size', 1000000, '.')
flags.DEFINE_integer('start_training', 2500, 'Number of training steps to start training.')
flags.DEFINE_integer('replay_ratio', 2, 'Number of updates per step.')
flags.DEFINE_string('env_name', 'cheetah-run', 'Environment name.')
FLAGS = flags.FLAGS


class flags_:
    #seed = 0
    max_steps = 1000
    batch_size = 128
    start_training = 1001
    replay_ratio = 2
    torch_deterministic = True
    eval_episodes = 5
    eval_interval = 5000
    env_name = 'cheetah-run'
#FLAGS = flags_()

def get_seed():
    return np.random.randint(0,1e8)

def get_done(termination, truncation):
        if not termination or truncation:
            done = 0.0
        else:
            done = 1.0
        return done

def sample_multibatch(buffer, batch_size, replay_ratio):
    batches = []
    for i in range(replay_ratio):
        batch = buffer.sample(batch_size)
        batches += [batch]
    return batches

def evaluate(eval_env, agent, eval_episodes: int, temperature: float = 0.0):
    returns = np.zeros(eval_episodes)
    for episode in range(eval_episodes):
        episode_done = False
        observation, _ = eval_env.reset(seed=get_seed())
        while episode_done is False:
            with torch.no_grad():
                action = agent.get_action(torch.from_numpy(observation).unsqueeze(0).to(agent.device), get_log_prob=False, temperature=0.0)
            action = action.detach().cpu().numpy()[0]
            next_observation, reward, termination, truncation, _ = eval_env.step(action)
            returns[episode] += reward
            observation = next_observation
            if termination or truncation:
                episode_done = True
    return {'returns': returns.mean()}
             
def log_to_wandb(step, infos):
    dict_to_log = {'timestep': step}
    for info_key in infos:
        dict_to_log[f'{info_key}'] = infos[info_key]
    wandb.log(dict_to_log, step=step)


def tmnf_env_setup(cfg):
    cfg = load_and_merge_platform(cfg)
    introscreen(cfg, askstart=secure_attribute_retrieval(lambda : cfg.ask_start, default=True))
    
    if cfg.vectorized.vectorize:
        tm_env = SB3Vectorized(n_envs = cfg.vectorized.n_envs, 
                               tracks=cfg.vectorized.tracks, 
                               cfg=cfg, obs_as_dict=False, step_parallel=True, 
                               lock = Lock())
    else:
        tm_env = CrashProofEnvironment(cfg)
        tm_env.init_environment()
    return tm_env
    

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "../configs",
    "config_name": "train.yaml",
}
@hydra.main(**_HYDRA_PARAMS)      
def main(cfg : TrainConfig, run_id : Optional[str] = None):
    SEED = get_seed()
    if False:
        wandb.init(
            config=FLAGS,
            entity='naumix',
            project='BRO_Torch',
            group=f'{FLAGS.env_name}',
            name=f'BRO_seed:{SEED}_RR:{FLAGS.replay_ratio}'
        )
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    env = eval_env = tmnf_env_setup(cfg)

    
    buffer = VecReplayBuffer(n_envs=cfg.vectorized.n_envs, buffer_size=FLAGS.max_steps, observation_size=env.observation_space.shape[-1], action_size=env.action_space.shape[-1], device=device)
    agent = BRO(env.observation_space.shape[-1], env.action_space.shape[-1], device=device, replay_ratio=FLAGS.replay_ratio, distributional=False)
    
    observation, _ = env.reset(seed=get_seed())
    for i in range(1, FLAGS.max_steps + 1):
        if i <= FLAGS.start_training:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = agent.get_action(torch.from_numpy(observation).unsqueeze(0).to(device), get_log_prob=False, temperature=1.0)
            action = action.detach().cpu().numpy()[0]
        next_observation, reward, termination, truncation, _ = env.step(action)
        done = get_done(termination, truncation)
        buffer.add(observation, next_observation, action, reward, done, {})
        observation = next_observation
        if termination or truncation:
            observation, _ = env.reset(seed=get_seed())
        if i > FLAGS.start_training:
            observations, next_observations, actions, rewards, dones = buffer.sample_multibatch(FLAGS.batch_size, FLAGS.replay_ratio)
            info = agent.update(i, observations, next_observations, actions, rewards, dones)
        if (i % FLAGS.eval_interval) == 0:
            eval_info = evaluate(eval_env, agent, FLAGS.eval_episodes)
            infos = {**info, **eval_info}
            log_to_wandb(i, infos)
            
if __name__ == '__main__':
    main()

        