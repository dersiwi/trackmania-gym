import numpy as np

ACTION_MAP = [
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

REVERSE_ACTION_MAP: dict[tuple[bool, bool, bool, bool], int] = {
    action: i for i, action in enumerate(ACTION_MAP)
}

def merge_action_sequences(seq1 : list[tuple[bool, bool, bool, bool]], seq2 : list[tuple[bool, bool, bool, bool]]):
    """Merges two lists containing discrete actions (i.e. 4-tuples of booleans.)"""
    larger, smaller = None, None
    if len(seq1) >= len(seq2):
        larger, smaller = seq1, seq2
    else:
        larger, smaller = seq2, seq1
    for i in range(len(smaller)):
        a1, a2, a3, a4 = smaller[i]
        b1, b2, b3, b4 = larger[i]
        larger[i] = (a1 or b1, a2 or b2, a3 or b3, a4 or b4)
    
    return larger

    
class ActionMode:
    DISCRETE = "discrete"
    CONTINUOUS_2D = "continuous_2d"
    CONTINUOUS_3D = "continuous_3d"
    CONTINUOUS_4D = "continuous_4d"


    class DiscreteIndexes:
        LEFT = 0
        RIGHT = 1
        ACCELERATION = 2
        BRAKE = 3

    class ContinuousIndexes:
        pass # TODO

    @staticmethod
    def is_valid(mode : str) -> bool:
        """Checks that given mode is a valid option"""
        return mode == ActionMode.DISCRETE or mode == ActionMode.CONTINUOUS_2D or mode == ActionMode.CONTINUOUS_3D or mode == ActionMode.CONTINUOUS_4D 
    
    @staticmethod
    def get_mode(continuous : bool, dim : int) -> int:
        """Returns the correct mode (may be used for action-generation); must pass config argumetns"""
        if not continuous:
            return ActionMode.DISCRETE
        if dim == 2: return ActionMode.CONTINUOUS_2D
        elif dim == 3: return ActionMode.CONTINUOUS_3D
        elif dim == 4: return ActionMode.CONTINUOUS_4D
        else: raise ValueError(f"No ActionMode implemented for continuous={continuous}, dim={dim}")
    
    @staticmethod
    def get_action_dim(mode : str):
        if mode == ActionMode.DISCRETE:
            return len(ACTION_MAP)
        elif mode == ActionMode.CONTINUOUS_2D:
            return 2
        elif mode == ActionMode.CONTINUOUS_3D:
            return 3
        elif mode == ActionMode.CONTINUOUS_4D:
            raise NotImplementedError("")
    
    @staticmethod
    def generate_random_action(mode : int, n_envs = 1, vectorized = False) -> np.ndarray | tuple[bool, bool, bool, bool]:
        action = None
        if mode == ActionMode.DISCRETE:
            action = np.random.randint(0, len(ACTION_MAP), size=(n_envs, ))
        else:
            action = np.random.random((n_envs, ActionMode.get_action_dim(mode))) * 2 - 1

        if not vectorized:
            return action[0]
        return action
  

    @staticmethod
    def transform_discrete(action : int) -> tuple[bool, bool, bool, bool]:
        """
        Transform action (index in ACTIONMAP) into action for ProcessWrapper
        
        :param action: index in ACTIONMAP
        :returns: tuple of booleans
        """
        return ACTION_MAP[action]


    @staticmethod
    def transform_steering(action : np.ndarray) -> int:
        """Transform a steering action € [-1, 1] into values used by the game, i.e. [-65536, 65536], which can then be passed to the game via steer-command."""
        steering_max_value = 65536 # max values for steering : https://donadigo.com/tminterface/commands
        return action * steering_max_value
    
    @staticmethod
    def transform_continuous_4d(action : np.ndarray) -> tuple[int, bool, bool]:
        raise NotImplementedError("4d action transform not implemented yet.")


    @staticmethod
    def produce_action_sequence(actionval : float, n_actions : int, discrete_idx : int, mappable_region = (-1, 1)) -> list[tuple[bool, bool, bool, bool]]:
        """Produces a sequence of n_actions interms of tuples (i.e. discrete) actions, mimicking the given action
        Args:
            action (float)  : The action to map from [-1, 1] to a list of tuple
            n_actions (int) : Number of actions to map to, i.e. length of list
            discrete_idx (int)  : What discrete action the given value corresponds to
            mappable_region (tuple[float, float]) : This region specifies what region of (-1, 1) is mapped; i.e. if you set this to -0.5, 0.5 then 
            everything < -0.5 is set to -1 and everything > 0.5 is also set to 1. Assert that lower bhound is smaller than upper bound. Always zero centerd.
        Returns:
            actionlist (list[tuple[bool]]) : A list of discrete actions
            
        Example:
            >>> produce_action_sequence(0.8, 5, 2, (0,1))
            [(False, False, True, False), (False, False, True, False), (False, False, True, False), (False, False, True, False), (False, False, False, False)]"""
        actions = [(False, False, False, False)] * n_actions
        stepsize = (mappable_region[1] - mappable_region[0]) / n_actions
        discrete_action = (0 == discrete_idx, 1 == discrete_idx, 2 == discrete_idx, 3 == discrete_idx)
        if actionval <= mappable_region[0]:
            return actions
        
        for i in range(n_actions):
            if actionval > mappable_region[0] + stepsize * i:
                actions[i] = discrete_action
        return actions

if __name__ == "__main__":
    for val in [-0.9, -0.7, -0.3, 0.0, 0.3, 0.9]:
        s = ""
        for t in ActionMode.produce_action_sequence(val, 5, 1, (-0.8, 0.8)):
            s += f" {t[1]}"
        print(s)

    print("NEXT")

    for val in [0.1, 0.3, 0.5, 0.7, 0.9]:
        s = ""
        for t in ActionMode.produce_action_sequence(val, 5, 1, (0,1)):
            s += f" {t[1]}"
        print(s)


    for j, i in zip([1,2,3], [3,4,43,2,21]):
        print(j,i)