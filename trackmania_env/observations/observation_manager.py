import numpy as np
import torch

from gymnasium import spaces
from tminterface.structs import SimStateData
from game_interaction.ipc_fields import IPCFields
from utils.image_converter import ImageConverter
from PIL import Image
import os
import traceback


class ObservationTerm:
    """This class contains observation terms for vecotr-like observations.
    Each observation-term returns a numpy-array of shape [N,]."""

    def __init__(self, dim : int, normalize : bool):
        self.dim = dim
        self.normalize = normalize
        self.env = None

    def set_env(self, env):
        from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
        self.env : TMNF_Single_Agent_Env = env
    
    def _get_obs(self, game_states : SimStateData, **kwargs) -> np.ndarray:
        raise NotImplementedError()
    
    def _normalize(self, obs) -> np.ndarray:
        raise NotImplementedError()

    def get_observation(self, game_states : SimStateData, **kwargs) -> np.ndarray:
        obs = self._get_obs(game_states, **kwargs)
        if self.normalize:
            obs = self._normalize(obs)
        return obs

class ObservationManager:

    class Colorspace:
        GRAYSCALE = 0
        RGB = 1
        BGRA = 2

        REV_DICT = {"grayscale" : 0, "rgb" : 1, "rgba" : 2} #this is somewhat ugly but this way the config contains a readable string

    def __init__(self, colorspace : str, convert_torch : bool, img_width : int, 
                 img_height : int, obs_have_img: bool = True, 
                 img_dump_freq : int = 1000000, n_dump_imgs : int= 20, normalize_obs : bool = False):
        """
        Parameters
        ---------
            - colorspace : string specifying the colorspace "grayscale", "rgb", "rgba"
            - convert_torch : if true, observations are converted from numpy to torch-tensor    
            - img_width     : width of image
            - img_height    : height of image
            - obs_have_img  : If True, image is passed along with state-vector, if false, only state vector is returned by get_observation(). If set to -1, no images are dumped.7
            - img_dump_freq : Specifies a frequency in which n_dump_imgs are dumped in logs/observations/. The idea behind this is to manually control, if the states produced by
            the environment match the expected. 
            - n_dump_imgs   : Amount of images that are dumped
        """
        self.obs_have_img = obs_have_img
        self.colorspace : int = ObservationManager.Colorspace.REV_DICT[colorspace]
        self.convert_torch : bool = convert_torch
        self.img_width = img_width
        self.img_height = img_height

        self.normalize_state_vec : bool = normalize_obs

        self.env = None
        self.n_channels = 1 if self.colorspace == ObservationManager.Colorspace.GRAYSCALE else 3

        self.info = {}

        self.convert_grayscale_to_uint8 = False

        self.img_dump_freq = img_dump_freq
        self.n_dump_imgs = n_dump_imgs
        self.n_imgs_dumped = 0
        self.img_dir = "logs/control_imgs"
        if self.img_dump_freq > -1:
            os.makedirs(self.img_dir, exist_ok=True)

        self.observation_terms : list[ObservationTerm] = []
        self.statevector_dim = 0

    def grayscaleimgs_as_uint8(self, val : bool):
        """Sets self.convert_grayscale_to_uint8 to given value."""
        self.convert_grayscale_to_uint8 = val

    def set_env(self, environment):
        from trackmania_env.envs.single_agent_env2 import TMNF_Single_Agent_Env
        self.env : TMNF_Single_Agent_Env = environment
        for term in self.observation_terms:
            term.set_env(environment)


    def get_values_from_state_dict(self, obs : SimStateData) -> np.ndarray:
        """This method gets a raw-observation simstate-obejcet (@param : obs) and converts it into a flattened
        vector, that is given to the feature-extractor network of the policy.
        
        Returns
        -------
        Vector of shape [N,] where N is the amount of non-image observation-fields."""
        obsarray = [term.get_observation(obs) for term in self.observation_terms]
        floatvec : np.ndarray = np.hstack(obsarray, dtype =  np.float32)
        
        assert floatvec.shape[0] == self.statevector_dim, f"Floatvector has size {floatvec.shape[0]}, however should be size {self.statevector_dim}"
        return floatvec
    

    def cnvt_imgs(self, images : np.ndarray) -> np.ndarray | torch.Tensor:
        """Converts image given by simulation into specified colortype and normalizes them into [0,1]."""
        if self.colorspace == ObservationManager.Colorspace.RGB:
            imgs = ImageConverter.bgra_to_rgb(images)
        elif self.colorspace == ObservationManager.Colorspace.GRAYSCALE:
            imgs = ImageConverter.bgra_to_graysacle(images, self.convert_grayscale_to_uint8)
        
        if self.convert_torch:
            imgs = torch.from_numpy(imgs)

        # only normlalize if not conversion, otherwiese it'll be stored as float again.
        if self.colorspace == ObservationManager.Colorspace.GRAYSCALE and self.convert_grayscale_to_uint8:
            return imgs
        else:
            return imgs / 255.0

    def get_observation_dict(self) -> spaces.Dict:
        """Returns observation dict for environment according to initialization.
        In most cases this is going to be a dictonray-observation space containing one 'image' field and one 'state'-field."""
        self.statevector_dim = 0
        for obsterm in self.observation_terms:
            self.statevector_dim += obsterm.dim
        return spaces.Dict({
                "image": spaces.Box(low=0, high=1, shape=(self.n_channels, self.img_height, self.img_width), dtype=np.uint8),
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(self.statevector_dim,), dtype=np.float32),
            })
        
    def reset(self):
        """Resets the observation-manager. This is called if the environment is reset."""
        pass

    def _dump_imgs(self, img : np.ndarray | torch.Tensor):
        """Dumps images periodically, according to self.img_dump_freq, self.n_imgs_dumped. This is to manually inspect the 
        output of the observations from the environment to the agent. Can be turned off by setting self.img_dump_ferq = -1."""
        try:
            ImageConverter.save_image(img, filepath=os.path.join(self.img_dir, f"processed_img_{self.env.total_steps}.png"))
        except Exception as e:
            traceback.print_exc()

        self.n_imgs_dumped += 1
        if self.n_imgs_dumped >= self.n_dump_imgs:
            self.n_imgs_dumped = 0
        

    
    def get_observation(self, raw_observation : dict[str, np.ndarray | SimStateData]) -> tuple[np.ndarray | dict[str, np.ndarray] | torch.Tensor | dict[str, torch.Tensor],dict[str,any]]:
        """
        Takes raw observations from TMInterface and dissects them into image.

        - Uses self.get_values_from_state_dict() to convert SimStateData into a state-observation vector
        - Uses self.normalize_state_vector() to normalize state vector observations, if specified
        - Uses self.cvt_igms() to convert the images into correct format, normalization and torch, if specified.

        Returns
        -------
            - dictionary ["img", "state"], if self.obs_have_img; otherwise it returns only a state-vector
            1) It converts both tensors to pytorch, if self.convert_torch, otherwise they are returned as numpy arrays.
            2) If self.normalize_state_vec it uses self.normalize_state_vector() to normalize the observations
        """
        state_observation_vector = self.get_values_from_state_dict(raw_observation[IPCFields.SIMSTATE])
            
        if self.convert_torch:
            state_observation_vector = torch.from_numpy(state_observation_vector)


        if self.obs_have_img:
            imgs = self.cnvt_imgs(raw_observation[IPCFields.IMG])
            if not self.img_dump_freq == -1 and (self.env.total_steps % self.img_dump_freq == 0 or self.n_imgs_dumped >= 1):
                self._dump_imgs(imgs)

            assert imgs.shape == (self.n_channels, self.img_height, self.img_width), f"Expected shape to be ({self.n_channels},{self.img_height}, {self.img_width}) but got {imgs.shape}"
            return {"image" : imgs, "state" : state_observation_vector},self.info
        else:
            return state_observation_vector,self.info