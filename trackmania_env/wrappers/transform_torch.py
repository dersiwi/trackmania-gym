import torch
import numpy as np
from gymnasium import ObservationWrapper
from typing import Dict
from bytefield import ByteArrayField,IntegerField,FloatField,BooleanField
from bytefield import unpack_bytes

class PytorchWrapper(ObservationWrapper):
    def __init__(self, env, dtype = torch.float):
        super().__init__(env)
        self.dtype = dtype 

    def get_SimStateData(self):
        """
        This function retrieves the SimStateData object from 
        the environment which this wrapper is put on top off.
        Since we dont know how many wrappers have been applied before this 
        wrapper here we recursively search for it.
        """
        field = self
        while True :
            if hasattr(field,"env"):
                field = getattr(field,"env")
            elif hasattr(field,"SimStateData"):
                return getattr(field,"SimStateData")
            else: 
                raise Exception("There is no SimStateData field in the wrapper chain")

    def observation(self,  observation : Dict[str,any],)-> Dict[str,torch.Tensor]:
        # this is neccessary to later retrived the values for objects from the bytefield module
        game_states = self.get_SimStateData()
        for k in observation:
            v = observation[k] 

            if isinstance(v, ByteArrayField): 
                value = v._getvalue(game_states)
                arr = list(value.to_bytearray())
                observation[k] = torch.tensor(arr,dtype=self.dtype)

            elif isinstance(v,(IntegerField,BooleanField,FloatField)):
                # this would also be valid unpack_bytes(game_states,v)
                observation[k] = torch.tensor(v._getvalue(game_states),dtype=self.dtype)

            elif isinstance(v,np.ndarray) and v.dtype == np.object_:
                arr = np.vstack(v).astype(np.float32)
                t = torch.from_numpy(arr)
                observation[k] = torch.tensor(t,dtype=self.dtype) 

            else:
                observation[k] = torch.tensor(v,dtype=self.dtype)

        return observation