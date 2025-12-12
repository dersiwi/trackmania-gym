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

class ActionMode:
    DISCRETE = 0
    CONTINUOUS_2D = 1
    CONTINUOUS_4D = 2

    class DiscreteIndexes:
        LEFT = 0
        RIGHT = 1
        ACCELERATION = 2
        BRAKE = 3

    class ContinuousIndexes:
        pass # TODO

    @staticmethod
    def is_valid(mode : int) -> bool:
        """Checks that given mode is a valid option"""
        return mode == ActionMode.DISCRETE or mode == ActionMode.CONTINUOUS_2D or mode == ActionMode.CONTINUOUS_4D 
    
    @staticmethod
    def transform_discrete(action : int) -> tuple[bool, bool, bool, bool]:
        """
        Transform action (index in ACTIONMAP) into action for ProcessWrapper
        
        :param action: index in ACTIONMAP
        :returns: tuple of booleans
        """
        return ACTION_MAP[action]
    
    @staticmethod
    def transform_continuous_2d(action : np.ndarray) -> tuple[int, bool, bool]:
        """
        Transform 2-d continuous action into a signal the process-wrapp can actually send to the envionrment. The action
        is expected to be within [-1, 1]. It is then mapped into steering, break, acceleration. steering is mapped as it is natively continous, 
        break and gas are mapped onto booleans, i.e. true or false.
        
        
        :param action: Description
        :type action: np.ndarray
        :returns: tuple of [steering, break, acceleration]
        """
        steering_max_value = 65536 # max values for steering : https://donadigo.com/tminterface/commands
        return action[0] * steering_max_value, action[1] <= 0, action[1] > 0
    
    @staticmethod
    def transform_continuous_4d(action : np.ndarray) -> tuple[int, bool, bool]:
        raise NotImplementedError("4d action transform not implemented yet.")
