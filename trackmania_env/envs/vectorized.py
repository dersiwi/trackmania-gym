import torch
import numpy as np
import gymnasium as gym
import random 
import time


from trackmania_env.envs.sec_env import CrashProofEnvironment
from trackmania_env.utils.spacetransform import SpaceTransformer
from configs.config import TrainConfig


class TrackAssignmentManager:
    # TODO - for the future 
    def __init__(self):
        pass


class VectorizedTMEnvironment(gym.Env):


    def __init__(self, n_envs : int, tracks : list[str], cfg : TrainConfig, obs_as_dict : bool = False,
                 alternation_between_tracks : bool = False, 
                    n_steps_per_track : int = 2048,
                    assign_rangom_track_at_init : bool = True,
                    assign_random_track_at_alternation : bool = False):
        """
        This instanciates a Vectorized TMNF-Environment. 

        Args:
            - n_envs (int)          : The number of environments
            - tracks (list[str])    : The tracks used by the single environments (this does not have to be the same amount as n_envs)
            - cfg (TraingConfig)    : Configuration file for the environments
            - obs_as_dict (bool)    : Boolean; Wheather obs are returned as dictionaries - or as singular blocks of arrays (can be transformed using SpaceTransformer later on)
            - n_steps_per_track (int)                   : Number of steps each environment performs on a track before alternating to the next
            - alternation_between_tracks (bool)         : If True, Each environment trains `n_steps_per_track` on each track, before switching to a new track
            - assign_rangom_track_at_init (bool)        : If Tracks are assigned randomly to each environment (at initialization), or in the order given by the list. 
            - assign_random_track_at_alternation (bool) : If Tracks are assigned randomly to each environment (after `n_steps_per_track`)
        
        """
        self.n_envs = n_envs
        self.tracks = tracks
        assert len(tracks) > 0, "You have to provide at least one track."
        self.cfg = cfg
        self.return_obs_as_dict : bool = obs_as_dict
        self.curr_track_id = [random.randint(0, len(tracks) - 1) if assign_rangom_track_at_init else 0 for i in range(n_envs)]
        """Stores the current track id for each environment. self.tracks[self.curr_track_id[i]] stores the track name for environment i."""
        self._steps_per_track = [[0 for i in range(len(self.tracks))] for i in range(self.n_envs)]
        """Stores the amount of steps each environment did in each track"""

        self.n_steps_per_track = n_steps_per_track
        self.assign_random_track_at_alternation = assign_random_track_at_alternation

        port = self.cfg.gmi.port
        self.envs : list[CrashProofEnvironment] = [CrashProofEnvironment(train_cfg=self.cfg, port = port+i, return_obs_as_dict = obs_as_dict) for i in range(self.n_envs)]
        for i in range(self.n_envs):
            self.envs[i].init_environment()
            self.envs[i].env.request_map(self.tracks[self.curr_track_id[i]])
        
        self.transformer = SpaceTransformer.get_instance()
        self.transformer.expect_vectorized(self.n_envs)
        self.obs_size = self.transformer.expected_dim

        self.total_steps = 0
        self.average_step_time = 0

        

    def step(self, action : torch.Tensor | np.ndarray):
        assert action.shape[0] == self.n_envs
        step_begin = time.time()
        info = []
        observationlist = []
        rewards = np.zeros((self.n_envs, ))
        terminated = np.zeros((self.n_envs, ))
        truncated = np.zeros((self.n_envs, ))

        for i in range(self.n_envs):
            obs, rew, term, trun, envinfo = self.envs[i].step(action[i])
            observationlist.append(obs)
            terminated[i] = term
            truncated[i] = trun
            rewards[i] = rew
            info.append(envinfo)
            
            if term or trun:
                o_res, info_res = self.envs[i].reset()
                observationlist[-1] = o_res
                info[-1]["terminal_observation"] = obs

            self._steps_per_track[i][self.curr_track_id[i]] += 1
        
        self.check_map_alternation()
        self.total_steps += 1
        self.average_step_time += (step_begin - self.average_step_time) / self.total_steps
        return self._stack_observations(observationlist), rewards, terminated, truncated, info
    
    def check_map_alternation(self) -> None:
        """Checks, whether an environment has made sufficiently many steps on its current map and if yes, it requests a new map."""
        if self._steps_per_track[0][self.curr_track_id[0]] % self.n_steps_per_track == 0:
            #alternate maps for all tracks, because we assume that all environments are synchronous in their steps
            for i in range(self.n_envs):
                self.curr_track_id[i] = (self.curr_track_id[i] + 1) % len(self.tracks) if not self.assign_random_track_at_alternation else random.randint(0, len(self.tracks) - 1) 
 
    def _stack_observations(self, observation_list : list[np.ndarray] | list[dict[str, np.ndarray]]) -> list[np.ndarray] | list[dict[str, np.ndarray]]:
        """
        This is to be implemented by a subclass of this environment. Depending on the returntype of the underlying single-environments 
        (i.e. CrashProofEnvironment -> SingleAgentEnv -> ObservationManager), this mrethod stacks the observations depending on if the observations
        are in dictionary-formt or in vector-format.
        """
        if self.return_obs_as_dict:
            return self._combine_dict_obs(observation_list)
        else:
            return np.hstack(observation_list)
        
    def _combine_dict_obs(self, observation_list : list[dict[str, np.ndarray]]) -> list[dict[str, np.ndarray]]:
        first = observation_list[0] # this method assumes all dictionaries have the same keys (they really should have.)
        return {key : np.stack([obs[key] for obs in observation_list], axis=0) for key in first.keys()}



    def reset(self, *, seed = None, options = None):
        observations = []
        infos = []
        for i in range(self.n_envs):
            observation, info = self.envs[i].reset()
            observations.append(observation)
            infos.append(info)
        return self._stack_observations(observations), infos
