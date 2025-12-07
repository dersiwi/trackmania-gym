import torch
import numpy as np
import gymnasium as gym
import random 
import time

from concurrent.futures import ThreadPoolExecutor

from trackmania_env.envs.sec_env import CrashProofEnvironment
from trackmania_env.utils.spacetransform import SpaceTransformer
from trackmania_env.utils.actionmap import ACTION_MAP

from configs.config import TrainConfig

class TrackAssignmentManager:
    # TODO - for the future 
    def __init__(self):
        pass


class VectorizedTMEnvironment(gym.Env):

    def __init__(self, n_envs : int, tracks : list[str], cfg : TrainConfig, obs_as_dict : bool = False,
                 step_parallel : bool = False,
                 alternation_between_tracks : bool = False, 
                    n_steps_per_track : int = 2048,
                    assign_rangom_track_at_init : bool = False,
                    assign_random_track_at_alternation : bool = False):
        """
        This instanciates a Vectorized TMNF-Environment. 

        Args:
            - n_envs (int)          : The number of environments
            - tracks (list[str])    : The tracks used by the single environments (this does not have to be the same amount as n_envs)
            - cfg (TraingConfig)    : Configuration file for the environments
            - step_parallel (bool)  : Steps each environment in parallel using a threadpool. If false, steps all environments in sequence.
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
        self.curr_track_id = [random.randint(0, len(tracks) - 1) if assign_rangom_track_at_init else i % len(tracks) for i in range(n_envs)]
        """Stores the current track id for each environment. self.tracks[self.curr_track_id[i]] stores the track name for environment i."""
        self._steps_per_track = [[0 for i in range(len(self.tracks))] for i in range(self.n_envs)]
        """Stores the amount of steps each environment did in each track"""

        self.step_parallel = step_parallel
        self.threadpool : ThreadPoolExecutor = None
        if self.step_parallel:
            self.threadpool = ThreadPoolExecutor(max_workers = self.n_envs)

        self.alternation_between_tracks = alternation_between_tracks
        self.n_steps_per_track = n_steps_per_track
        self.assign_random_track_at_alternation = assign_random_track_at_alternation

        port = self.cfg.gmi.port
        self.envs : list[CrashProofEnvironment] = [CrashProofEnvironment(train_cfg=self.cfg, port = port+i, return_obs_as_dict = obs_as_dict) for i in range(self.n_envs)]
        
        # using a threadpool here is much much faster than initializing sequentially, as innit_environment also starts the game
        with ThreadPoolExecutor(max_workers=self.n_envs) as executor:
            _ = executor.map(lambda env : env.init_environment(), self.envs)

        for i in range(self.n_envs):
            self.envs[i].env.request_map(self.tracks[self.curr_track_id[i]])

        #self.observation_space = self._build_observation_space()
        self.observation_space = self.envs[0].observation_space
        self.action_space = gym.spaces.Discrete(len(ACTION_MAP))
        
        self.transformer = SpaceTransformer.get_instance()
        self.transformer.expect_vectorized(self.n_envs)
        self.obs_size = self.transformer.expected_dim

        self.total_steps = 0
        self.average_step_time = 0


    def _build_observation_space(self) -> gym.spaces.Space:
        """Builds the observation space for the vectorized environment."""
        if self.return_obs_as_dict:
            spacedict = {}
            for term in self.envs[0].obs_manager.terms:
                termspace = term.observation_space
                vectorized_shape = tuple([self.n_envs] + list(termspace.shape))
                low, high = termspace.low.flat[0], termspace.high.flat[0]   # NOTE : This is not correct; but for most terms it should be fine.
                spacedict[term.name] = gym.spaces.Box(low, high, vectorized_shape)
            return gym.spaces.Dict(spacedict)
        else:
            return gym.spaces.Box(-np.inf, np.inf, (self.n_envs, self.obs_size))

    def _step_env(self, i, action):
        """This is a helper method used by the Threadpools, if steppig is executed in parallel.
        """
        return i, self.envs[i].step(action)
    

    def step(self, action : torch.Tensor | np.ndarray):
        assert action.shape[0] == self.n_envs
        step_begin = time.time()

        info = []
        observationlist = []
        rewards = np.zeros((self.n_envs,))
        terminated = np.zeros((self.n_envs,), dtype=bool)
        truncated = np.zeros((self.n_envs,), dtype=bool)

        # start the 'parallel processing' of environments in the threadpool
        if self.step_parallel:
            futures = [self.threadpool.submit(self._step_env, i, action[i]) for i in range(self.n_envs)]

        results = [None] * self.n_envs

        # collect futures and store in results
        if self.step_parallel:
            for f in futures:
                idx, result = f.result()
                results[idx] = result

        for i, result_tuple in enumerate(results):
            # collect results or step environment, process sequentially.
            if self.step_parallel:
                (obs, rew, term, trun, envinfo) = result_tuple
            else:
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
        self.average_step_time += (time.time() - step_begin - self.average_step_time) / self.total_steps
        print(self.average_step_time)
        return self._stack_observations(observationlist), rewards, terminated, truncated, info
    

    def finalize_process(self, **kwargs) -> None:
        """Stops execution for all environments."""
        for env in self.envs:
            env.finalize_process(reinit=False)
    
    def check_map_alternation(self) -> None:
        """Checks, whether an environment has made sufficiently many steps on its current map and if yes, it requests a new map."""
        if self.alternation_between_tracks and self._steps_per_track[0][self.curr_track_id[0]] % self.n_steps_per_track == 0:
            #alternate maps for all tracks, because we assume that all environments are synchronous in their steps
            for i in range(self.n_envs):
                self.curr_track_id[i] = (self.curr_track_id[i] + 1) % len(self.tracks) if not self.assign_random_track_at_alternation else random.randint(0, len(self.tracks) - 1)
                self.envs[i].env.request_map(self.tracks[self.curr_track_id[i]])
            time.sleep(10)  #<-- TODO : This is very crude and not scalable (i think) - the problem is that the learner wants to step before the track has been changed, resulting in answreed requests form the environment to the game
 
    def _stack_observations(self, observation_list : list[np.ndarray] | list[dict[str, np.ndarray]]) -> list[np.ndarray] | list[dict[str, np.ndarray]]:
        """
        This is to be implemented by a subclass of this environment. Depending on the returntype of the underlying single-environments 
        (i.e. CrashProofEnvironment -> SingleAgentEnv -> ObservationManager), this mrethod stacks the observations depending on if the observations
        are in dictionary-formt or in vector-format.
        """
        if self.return_obs_as_dict:
            return self._combine_dict_obs(observation_list)
        else:
            return np.stack(observation_list, axis=0)
        
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



from stable_baselines3.common.vec_env import VecEnv

class SB3Vectorized(VecEnv):

    def __init__(self, n_envs : int, tracks : list[str], cfg : TrainConfig, obs_as_dict : bool = False, step_parallel : bool = False,
                 alternation_between_tracks : bool = False, 
                    n_steps_per_track : int = 2048,
                    assign_rangom_track_at_init : bool = False,
                    assign_random_track_at_alternation : bool = False):
        self.vecenv = VectorizedTMEnvironment(n_envs, tracks, cfg, obs_as_dict, step_parallel, alternation_between_tracks, n_steps_per_track, 
                                              assign_rangom_track_at_init, assign_random_track_at_alternation)
        super().__init__(n_envs, self.vecenv.envs[0].observation_space, self.vecenv.envs[0].action_space)
        self.actions = None
    def reset(self):
        obs, infos = self.vecenv.reset()
        return obs
    
    def step_async(self, actions):
        self.actions = actions
    
    def step_wait(self):
        obs, rewards, terminated, truncated, info = self.vecenv.step(self.actions)
        dones = np.logical_or(terminated, truncated)
        return obs, rewards, dones, info
    
    def close(self):
        self.vecenv.finalize_process()
    
    def seed(self, seed = None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """
        Call `method_name` for each underlying env.
        Returns list of results.
        """
        if indices is None:
            indices = range(self.num_envs)
        if isinstance(indices, int):
            indices = [indices]

        results = []
        for i in indices:
            env = self.vecenv.envs[i]
            method = getattr(env, method_name)
            results.append(method(*method_args, **method_kwargs))
        return results

    def get_attr(self, attr_name, indices=None):
        """
        Get attribute from underlying envs.
        """
        if indices is None:
            indices = range(self.num_envs)
        if isinstance(indices, int):
            indices = [indices]

        results = []
        for i in indices:
            env = self.vecenv.envs[i]
            results.append(getattr(env, attr_name))
        return results

    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute on underlying envs.
        """
        if indices is None:
            indices = range(self.num_envs)
        if isinstance(indices, int):
            indices = [indices]

        for i in indices:
            env = self.vecenv.envs[i]
            setattr(env, attr_name, value)

    def env_is_wrapped(self, wrapper_class, indices=None):
        """
        Return True/False for each env: is it wrapped by wrapper_class?
        """
        if indices is None:
            indices = range(self.num_envs)
        if isinstance(indices, int):
            indices = [indices]

        results = []
        for i in indices:
            env = self.vecenv.envs[i]
            wrapped = False
            current = env
            # unroll wrappers
            while hasattr(current, "env"):
                if isinstance(current, wrapper_class):
                    wrapped = True
                    break
                current = current.env
            # last check (if not using classic wrappers)
            if isinstance(current, wrapper_class):
                wrapped = True
            results.append(wrapped)
        return results
