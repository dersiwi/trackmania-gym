import torch
from gymnasium import ObservationWrapper
from typing import Dict
from bytefield import ByteArrayField

class PytorchWrapper(ObservationWrapper):
    def __init__(self, env, dtype = torch.float):
        super().__init__(env)
        self.dtype = dtype 

    def observation(self,  observation : Dict[str,any],)-> Dict[str,torch.Tensor]:
        for key in  observation:
            #TODO need to figure out how to convert ByteArrayField to tensor
            # one idea woudl be to call _getvalue() and then on the returned object to_bytearray()  
            if isinstance(observation[key], ByteArrayField): 
                raise Exception("Converting ByteArrayField to tensor is not yet implemented ")
                observation[key] = torch.tensor(list(observation[key]),dtype=self.dtype) 
            else:
                observation[key] = torch.tensor(observation[key],dtype=self.dtype)
            # this is necessary so we dont have to add dimensions in the forwards functions off all models
            observation[key] = observation[key].unsqueeze(0) if len(observation[key].shape) == 0 else observation[key]
        return observation