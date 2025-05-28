import functools
from typing import List,Dict
import gymnasium as gym 
from gymnasium import ObservationWrapper
from simstate_space_dict import simstate_space_dict
from tminterface.structs import SimStateData

class ObservationFilter(ObservationWrapper):
    def __init__(self, env,observations_list):
        super().__init__(env)
        self.observation_list = observations_list
        self.observation_space = self.make_gym_space_dict(observations_list) 

    # step() and reset() are unnecessary to implement since the ObservationWrapper automatically defines them like that 
    # this is only for people to understand what happens under the hood 
    """
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
    """

    def observation(self,observation : Dict[str,any]):
        """This observation method assumes that the observation is a whole SimStateData object and
         filters only the wanted fields. """
        obs = {}
        for s in self.observation_list:
            if s == "image": 
                obs[s] = observation[s]
            else: 
                obs[s] = self.get_value_SimStateData(s,observation["SimStateData"])
        return obs
    
    def _get_obs(self)-> Dict:
        """This function should actually never be used for learning or inference 
        since we got the observation function. Mainly use it for logging or debugging"""
        return self.observation(self.env._get_obs())

    def make_gym_space_dict(self,observations_list:List[str])-> gym.spaces.Dict:
        """This method returns a dictionary which holds the gym.spaces objects for 
        the corresponding key defined in observations_list. The strings from 
        observations_list have to be the same as in simstate_space_dict."""
        gym_dict = {}
        for key in observations_list:
            if key in simstate_space_dict:
                gym_dict[key] = simstate_space_dict[key]
            else :
                raise KeyError(f"Key '{key}' not found in simstate_space_dict.")
        return gym.spaces.Dict(gym_dict)

    # from https://discuss.python.org/t/enhancing-getattr-to-support-nested-attribute-access-with-dotted-strings/74305/9
    def get_value_SimStateData(self,key:str,data: SimStateData) -> any :
        return functools.reduce(getattr, key.split('.'), data)