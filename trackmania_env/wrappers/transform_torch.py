import torch
import numpy as np
from gymnasium import ObservationWrapper
from typing import Dict
from bytefield import ByteArrayField
class PytorchWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self,  observation : Dict[str,any]):
        for key in  observation:
            if isinstance(observation[key], ByteArrayField): 
                y = observation[key].byte_array
                z = list(y)
                print(z)
                observation[key] = torch.tensor(list(observation[key])) 
            else:
                observation[key] = torch.tensor(observation[key])
        return observation