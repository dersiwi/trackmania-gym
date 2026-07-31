
from tminterface.structs import SimStateData

from trackmania_gym.trackmania_env.rewards.reward_calculation import RewradCalculator
from trackmania_gym.trackmania_env.rewards.reward_terms.basic_terms import DriveForwardReward
from trackmania_gym.game_interaction.ipc_fields import IPCFields

class ForwardReward(RewradCalculator):

    def __init__(self, drive_forward_reward,
                normalize : bool = False,
                **kwargs ):
        """
        Initializes the reward manager with explicit reward weights and parameters
        for modeling centerline distance reward using a Gaussian function.

        Args:
            drive_forward_reward (float):   Weight for forward-acceleration acitons
    """
        
        super().__init__(normalize)

        self.terms = [DriveForwardReward(drive_forward_reward)]
        if len(kwargs) > 0:
            print(f"Got additional kwargsuments that are not used; ignoring them : {kwargs.keys()}. Maybe they're used by a class that inherits.")

    def reset(self):
        self.current_refline_idx = 0
        return super().reset()