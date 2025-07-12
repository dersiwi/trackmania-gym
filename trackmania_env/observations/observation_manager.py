import numpy as np
import torch
import functools

from gymnasium import spaces
from tminterface.structs import SimStateData
from game_interaction.ipc_fields import IPCFields
from utils.image_converter import ImageConverter
from utils.box_dimension_flattend import get_flattened_dict_dim
from simstate_space_dict import simstate_space_dict

from bytefield import ByteArrayField,IntegerField,FloatField,BooleanField, ByteStruct

class ObservationManager:

    class Colorspace:
        GRAYSCALE = 0
        RGB = 1
        BGRA = 2

        REV_DICT = {"grayscale" : 0, "rgb" : 1, "rgba" : 2} #this is somewhat ugly but this way the config contains a readable string

    def __init__(self, observation_list : list[str], colorspace : str, convert_torch : bool, img_width : int, img_height : int):
        self.observation_list : list[str] = observation_list
        self.obs_have_img : bool = "image" in observation_list
        self.colorspace : int = ObservationManager.Colorspace.REV_DICT[colorspace]
        self.convert_torch : bool = convert_torch
        self.img_width = img_width
        self.img_height = img_height

        self.env = None
        self.n_channels = 1 if self.colorspace == ObservationManager.Colorspace.GRAYSCALE else 3

        self.info = {}

        if self.obs_have_img:
            self.observation_list.remove("image")

    def set_env(self, environment):
        from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
        self.env : TMNF_Single_Agent_Env = environment


    def get_values_from_state_dict(self, obs : SimStateData) -> np.ndarray:
        """This method gets a raw-observation simstate-obejcet (@param : obs) and converts it into a flattened
        vector, that is given to the feature-extractor network of the policy.
        
        Returns
        -------
        Vector of shape [N,] where N is the amount of non-image observation-fields."""
        return self._filter_and_flatten_from_list(obs)
    
    def _filter_and_flatten_from_list(self, simstatedata : SimStateData) -> np.ndarray:
        """This observation method assumes that the observation is a whole SimStateData object and
         filters only the wanted fields. """
        flat_parts = []
        for s in self.observation_list:
            value = self.__convert_field_to_numpy_value(self.get_value_SimStateData(s, simstatedata), simstatedata)
            flat_parts.append(self.__flatten_single_obs(obsname = s, value = value)) 
        return np.concatenate(flat_parts, axis=0)
    

    def __flatten_single_obs(self, obsname: any, value : any) -> np.ndarray:
        subspace = simstate_space_dict[obsname]
        if isinstance(subspace, spaces.Box):
            flat = np.asarray(value, dtype=subspace.dtype).flatten()

        elif isinstance(subspace, spaces.Discrete):
            # Convert integer into one-hot or just the integer (here: integer)
            flat = np.array([value], dtype=np.int64)

        elif isinstance(subspace, spaces.MultiBinary) or isinstance(subspace, spaces.MultiDiscrete):
            flat = np.asarray(value, dtype=np.int64).flatten()

        else:
            raise NotImplementedError(f"Unsupported space type for key '{obsname}': {type(subspace)}")

        return flat
    
    def __convert_field_to_numpy_value(self, field, state : SimStateData) -> np.ndarray:
        if isinstance(field, ByteArrayField): 
            value : ByteStruct = field._getvalue(state)
            value = list(value.to_bytearray())

        elif isinstance(field,(IntegerField,BooleanField,FloatField)):
            # this would also be valid unpack_bytes(game_states,v)
            value = field._getvalue(state)

        elif isinstance(field,np.ndarray) and field.dtype == np.object_:
            arr = np.vstack(field).astype(np.float32)
            value = torch.from_numpy(arr)

        else:
            value = field
        return value

    

    def get_observation_dict(self) -> spaces.Dict:
        """Returns observation dict for environment according to initialization."""
        statevector_dim = get_flattened_dict_dim([simstate_space_dict[obsname] for obsname in self.observation_list])

        return spaces.Dict({
                "image": spaces.Box(low=0, high=1.0, shape=(self.n_channels, self.img_height, self.img_width), dtype=np.float32),
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(statevector_dim,), dtype=np.float32),
            })
    
    def reset(self):
        pass

    
    # from https://discuss.python.org/t/enhancing-getattr-to-support-nested-attribute-access-with-dotted-strings/74305/9
    def get_value_SimStateData(self,key:str,data: SimStateData) -> any :
        return functools.reduce(getattr, key.split('.'), data)

    def cnvt_imgs(self, images : np.ndarray) -> np.ndarray | torch.Tensor:

        if self.colorspace == ObservationManager.Colorspace.RGB:
            imgs = ImageConverter.bgra_to_rgb(images)
        elif self.colorspace == ObservationManager.Colorspace.GRAYSCALE:
            imgs = ImageConverter.bgra_to_graysacle(images)
        
        if self.convert_torch:
            imgs = torch.from_numpy(imgs)

        return imgs / 255
    
    def get_observation(self, raw_observation : dict[str, np.ndarray | SimStateData]) -> tuple[np.ndarray | dict[str, np.ndarray] | torch.Tensor | dict[str, torch.Tensor],dict[str,any]]:
        """
        Takes raw observations from TMInterface and dissects them into image
        """
        state_observation_vector = self.get_values_from_state_dict(raw_observation[IPCFields.SIMSTATE])
        if self.convert_torch:
            state_observation_vector = torch.from_numpy(state_observation_vector)


        if self.obs_have_img:
            imgs = self.cnvt_imgs(raw_observation[IPCFields.IMG])
            assert imgs.shape == (self.n_channels, self.img_height, self.img_width), f"Expected shape to be ({self.n_channels},{self.img_height}, {self.img_width}) but got {imgs.shape}"
            return {"image" : imgs, "state" : state_observation_vector},self.info
        else:
            return state_observation_vector,self.info