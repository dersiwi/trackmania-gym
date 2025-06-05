from typing import Dict
import torch
from gymnasium import ObservationWrapper
from gymnasium.vector import VectorObservationWrapper

class PytorchWrapper(ObservationWrapper):
    def __init__(self, env, dtype = torch.float):
        super().__init__(env)
        self.dtype = dtype 
        self.obs = {}
    def observation(self,  observation : Dict[str,any],)-> Dict[str,torch.Tensor]:
        for k,v in observation.items():
            self.obs[k] = torch.tensor(v,dtype=self.dtype)
        return self.obs

class Vec_PytorchWrapper(VectorObservationWrapper):
    def __init__(self, env, dtype = torch.float):
        super().__init__(env)
        self.dtype = dtype 
        self.obs = {}
    def observation(self,  observation : Dict[str,any],)-> Dict[str,torch.Tensor]:
        for k,v in observation.items():
            self.obs[k] = torch.tensor(v,dtype=self.dtype)
        return self.obs