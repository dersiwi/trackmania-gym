"""
Custom Gymnasium Environment 
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
"""
from typing import Any,Dict,Tuple,Optional
import gymnasium as gym
import numpy as np
import time 
from game_interaction.tminterface2 import TMInterface, MessageType
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
            user_profile : int,
            color_channels : int = 3
            ):
        """
        Initializes the custom Gymnasium environment.
        This constructor sets up the basic structure of the environment.
        As required by Gymnasium environments, it defines the action and observation spaces.
        We also define some other important varibales in order to communicate with TMInterface
        """
        self.img_width = img_width
        self.img_height = img_height
        self.color_channels= color_channels
        self.img_shape = (self.color_channels,img_height,img_width)
        
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
        # (left, right, accelerate, brake)
        # 0 Forward
        (False,False,True,False), 
        # 1 Forward left
        (True,False,True,False),
        # 2 Forward right
        (False,True,True,False),
        # 3 Nothing
        (False,False,False,False),
        # 4 Nothing left
        (True,False,False,False),
        # 5 Nothing right
        (False,True,False,False),
        # 6 Brake
        (False,False,False,True),
        # 7 Brake left
        (True,False,False,True),
        # 8 Brake right
        (False,True,False,True),
        # 9 Brake and accelerate
        (False,False,True,True),
        # 10 Brake and accelerate left
        (True,False,True,True),
        # 11 Brake and accelerate right
        (False,True,True,True),
        ]
        self.action_space = gym.spaces.Discrete(len(self.action_map))
        
        # TODO i dont know if we should start the game and everything here or if we put in a somehwat controller class
        self.gim.launch_game(timeout = 20)
        while not self.gim.is_game_running(): time.sleep(0)
        
        gim.register_iface()
        self.tmi : TMInterface = self.gim.get_tminterface()
        
        _msgtype = self.tmi._read_int32()
        self.tmi.on_connect_event(user_profile=self.user_profile,map_to_load=self.map_to_load)
        self.tmi._respond_to_call(MessageType.SC_ON_CONNECT_SYNC)
        
    def _get_info(self) -> Dict[str,Any]:
        """
        Helper function for computing additional information (e.g. for debugging or logging)
        """
        info = {} 
        return info
    
    def _get_obs(self) -> Dict:
        """
        Helper function to translate the the environment's state into an observation
        """
        image = None
        game_states = None
       
        while True:
            msgtype = self.tmi._read_int32()
            
            if (image is not None) and (game_states is not None): break

            elif msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                _ = self.tmi._read_int32()  # Discard simulation time
                game_states = self.tmi.get_simulation_state() 
                # Request Frame
                self.tmi.request_frame(self.img_width, self.img_height)
                self.tmi._respond_to_call(msgtype)

            elif msgtype == int(MessageType.SC_REQUESTED_FRAME_SYNC):
                image = self.tmi.get_frame(self.img_width, self.img_height) # this is apparently in BRGA
                self.tmi._respond_to_call(msgtype)

            elif msgtype == int(MessageType.C_SHUTDOWN):
                self.tmi.close()
                break

            else:
                # Acknowledge all other messages but ignore their contents
                self.tmi._respond_to_call(msgtype)
        
        assert (image is not None) and (game_states is not None)
        """
        TODO should the BRGA be converted into another color space ?
        And should this happen here or before we feed it to the NN
        """
        # TODO we need here a somewhat filter method to only get the relevant state informatios from the engine  
            # Extract relevant parts of game_states (mocked here)
        gear = game_states.scene_mobil.engine.gear
        speed = game_states.scene_mobil.max_linear_speed
        burnout_state = game_states.scene_mobil.burnout_state  
        velocity = game_states.velocity
        observations = {
        "image": image,
        "gear": gear,
        "max_linear_speed": speed,
        "burnout_state": burnout_state,
        "velocity": velocity
        }
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
        
        (left,right,accelerate,brake) = self.action_map[action]
        game_states = None 
        image = None
        observation = None
        # TODO send the action via tminterface here
        self.tmi._respond_to_call(MessageType.SC_RUN_STEP_SYNC)
        while True:
            msgtype = self.tmi._read_int32()
        
        # ============================================= READ INCOMING MESSAGES
            if (image is not None): break #and (game_states is not None): break
                
            elif msgtype == int(MessageType.SC_RUN_STEP_SYNC): # simulation step is complete

            # ============================ BEGIN ON RUN STEP ============================

                self.tmi.request_frame(self.img_width, self.img_height)
                self.tmi.set_input_state(left, right, accelerate, brake)
                self.tmi.set_speed(1)
                #game_states = self.tmi.get_simulation_state()
                
            # ============================ END ON RUN STEP ============================
                self.tmi._respond_to_call(msgtype)

            elif msgtype == int(MessageType.SC_REQUESTED_FRAME_SYNC):
                image = self.tmi.get_frame(self.img_width, self.img_height) # this is apparently in BRGA
                self.tmi._respond_to_call(msgtype)
    
    
            elif msgtype == int(MessageType.C_SHUTDOWN):
                self.tmi.close()
            else:
                # Acknowledge all other messages but ignore their contents
                self.tmi._respond_to_call(msgtype)
        
        # TODO recieve data from the simulator via tmi. NEED TO RETHINK THIS WHOLE MESS
        #observation =  self._get_obs(); 
        #gear = game_states.scene_mobil.engine.gear
        #speed = game_states.scene_mobil.max_linear_speed
        #burnout_state = game_states.scene_mobil.burnout_state  
        #velocity = game_states.velocity
        #observation = {
        #"image": image,
        #"gear": gear,
        #"max_linear_speed": speed,
        #"burnout_state": burnout_state,
        #"velocity": velocity
        #}
        # TODO check if episode terminated and or tuncated
        # terminated if reached the goal, truncated means that a timelimit has been reached but MDP is not in a terminal state 
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