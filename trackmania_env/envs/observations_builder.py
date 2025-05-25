from typing import List,Dict
from simstate_space_dict import simstate_space_dict
import gymnasium as gym 
import functools
from tminterface.structs import SimStateData

def make_gym_space_dict(observations_list:List[str])-> gym.spaces.Dict:
    """This method returns a dictionary which holds the gym.spaces objects for 
    the corresponding key defined in observations_list. The strings from 
    observations_list have to be the same as in simstate_space_dict"""
    gym_dict = {}
    for key in observations_list:
        if key in simstate_space_dict:
            gym_dict[key] = simstate_space_dict[key]
        else :
            raise KeyError(f"Key '{key}' not found in simstate_space_dict.")
    return gym.spaces.Dict(gym_dict)

# from https://discuss.python.org/t/enhancing-getattr-to-support-nested-attribute-access-with-dotted-strings/74305/9
def deep_getattr(obj, attr):
    return functools.reduce(getattr, attr.split('.'), obj)

def get_value_SimStateData(key:str,data: SimStateData) -> any :
    return deep_getattr(obj= data,attr=key)


    


        