"""
Custom Gymnasium Environment 
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
"""
from typing import Any,Dict,Tuple,Optional
import gymnasium as gym
import numpy as np

from game_interaction.tminterface2 import TMInterface
from game_interaction.game_instance_manager2 import GameInstanceManager

class TMNF_Single_Agent_Env(gym.Env):
    """The reinforcement learning environment for Trackmania Nations Forever"""

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}
    
    # init, step and reset have to be implemented for the class to be gym compatible

    def __init__(
            self,
            img_width: int,
            img_height: int,
            port: str,
            observations_space: gym.spaces.Dict,
            gim: GameInstanceManager,
            map_to_load : str,
            user_profile : int
            ):
        """
        Initializes the custom Gymnasium environment.
        This constructor sets up the basic structure of the environment.
        As required by Gymnasium environments, it defines the action and observation spaces.
        We also define some other important varibales in order to communicate with TMInterface
        """
        self.img_width = img_width
        self.img_height = img_height
        self.img_shape = (3,img_height,img_width)
        
        self.port = port
        self.gim = gim 
        self.map_to_load = map_to_load
        self.user_profile = user_profile
        self.observation_space = observations_space
        
        """
        these are the inputs we will later give to the game engine ,
        copied from linesightrl /config_files/inputs_list.py
        """
        self.action_map = [
        # 0 Forward
        {"left": False,"right": False,"accelerate": True,"brake": False,}, 
        # 1 Forward left
        {"left": True,"right": False,"accelerate": True,"brake": False,},
        # 2 Forward right
        {"left": False,"right": True,"accelerate": True,"brake": False,},
        # 3 Nothing
        {"left": False,"right": False,"accelerate": False,"brake": False,},
        # 4 Nothing left
        {"left": True,"right": False,"accelerate": False,"brake": False,},
        # 5 Nothing right
        {"left": False,"right": True,"accelerate": False,"brake": False,},
        # 6 Brake
        {"left": False,"right": False,"accelerate": False,"brake": True,},
        # 7 Brake left
        {"left": True,"right": False,"accelerate": False,"brake": True,},
        # 8 Brake right
        {"left": False,"right": True,"accelerate": False,"brake": True,},
        # 9 Brake and accelerate
        {"left": False,"right": False,"accelerate": True,"brake": True,},
        # 10 Brake and accelerate left
        {"left": True,"right": False,"accelerate": True,"brake": True,},
        # 11 Brake and accelerate right
        {"left": False,"right": True,"accelerate": True,"brake": True,},
        ]
        self.action_space = gym.spaces.Discrete(len(self.action_map))
        
        # TODO i dont know if we should start the game and everything here or if we put in a somehwat controller class
        self.gim.launch_game(timeout = 20)
        gim.register_iface()
        self.tmi : TMInterface = self.gim.get_tminterface()
        msgtype = self.tmi._read_int32()
        self.tmi.on_connect_event(user_profile=self.user_profile,map_to_load=self.map_to_load)
        
    def _get_info(self) -> Dict[str,Any]:
        """
        Helper function for computing additional information (e.g. for debugging or logging)
        """
        info = {} 
        return info
    
    def _get_obs(self) -> gym.spaces.Dict:
        """
        Helper function to translate the the environment's state into an observation
        """
        observations = None 
        return observations

    def step(self, action) -> Tuple[gym.spaces.Dict,float,bool,bool,Dict[str,Any]]:
        """
        The `step` function is a core component of the Gymnasium environment API and contains 
        the main game logic, including state transitions and reward calculations.
        This method applies the specified action to the environment, updates the 
        internal state accordingly, and returns the results of that action.

        Parameters:
            action (int): The action that should be performed

        Returns:
        observation (gym.spaces.Dict): The initial observation of the environment.
        reward (float): The scalar reward signal for the agent's action.
        terminated (bool): Whether the episode has ended because the task is successfully 
                           completed (e.g., agent reached the goal).
        truncated (bool): Whether the episode ended due to a time limit or other external cutoff.
        info (dict): A dictionary with additional information (e.g. for debugging or logging).
        """
        input_to_tmi = self.action_map[action]
        # TODO send the action via tminterface here
        
        # TODO recieve data from the simulator via tmi
        observation =  self._get_obs(); 
           
        # TODO check if episode terminated and or tuncated
        # terminated if reached the goal, truncated measn that a timelimit has been reached but MDP is not in a terminal state 
        terminated = False
        truncated = False
        
        # TODO calculate reward
        reward = 0.
        
        # TODO also store some info for logging or debugging
        info = self._get_info() 
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed = None, options = None)-> Tuple[gym.spaces.Dict,Dict[str,Any]]:
        """
        Resets the environment to start a new episode.

        Parameters:
            seed (int, optional): Seed for the random number generator to ensure 
                              deterministic behavior. Used by calling `super().reset(seed=seed)`.
            options (dict, optional): Additional options for resetting the environment. 
                                    These can be used to modify behavior at reset time.

        Returns:
            observation (gym.spaces.Dict): The initial observation of the environment.
            info (dict): Additional information that may be useful for debugging or analysis.
    
        This method is required by the Gymnasium API and is called at the beginning of each new episode.
        It ensures the environment starts in a consistent and valid state.
        """
        super().reset(seed=seed, options=options)
        """
        TODO here we can spawn the car (agent) on the same start positon or somewhere else 
        (one of the TM Youtube channels said that random spawning helps)
        """
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def render(self, mode = "human") -> Optional[np.array]:
        """
        This defines the render method. It supports:
            - mode="human": The environment is continuously rendered in the current 
            display or terminal, usually for human consumption.
            
            - mode="rgb_array": Return a single frame representing the current state
            of the environment. A frame is a np.ndarray with shape (x, y, 3) .
        """
        if mode == "human" :
            # TODO should we actually do here some render things ?
            return None
        if mode == "rgb_array":
            # TODO retun the actual frame
            return np.zeros(shape = self.img_shape)