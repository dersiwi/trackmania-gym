
from tminterface.structs import CheckpointData, SimStateData, CheckpointTime

from trackmania_env.rewards.reward_calculation import RewradCalculator
from trackmania_env.rewards.reward_terms.basic_terms import DriveForwardReward
from game_interaction.ipc_fields import IPCFields
import numpy as np
from trackmania_env.rewards.reward_terms.basic_terms import AccumulatedDistanceReward, RaceFinished
from trackmania_env.rewards.reward_terms.tracknormalized import AccumulatedTotal


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
    

class TrackNormalized(RewradCalculator):
    
    def __init__(self, 
                 accum_distance_weight: float,
                 total_reward : float,
                accum_enhanced_by_amount_travelled: bool,
                accum_exp_factor: float,
                race_finished_weight :float,
                normalize : bool = False,
                **kwargs ):
        """
        Initializes the reward manager with explicit reward weights and parameters
        for modeling centerline distance reward using a Gaussian function.

        Args:
            drive_forward_reward (float):   Weight for forward-acceleration acitons
    """
        
        super().__init__(normalize)

        self.terms = [AccumulatedTotal(
                accum_distance_weight,
                total_reward = total_reward,
                enhanced_by_amount_travelled=accum_enhanced_by_amount_travelled,
                exponential_factor=accum_exp_factor,
            ),
            RaceFinished(weight=race_finished_weight)]
        if len(kwargs) > 0:
            print(f"Got additional kwargsuments that are not used; ignoring them : {kwargs.keys()}. Maybe they're used by a class that inherits.")

class AccumReward(RewradCalculator):
    
    def __init__(self, 
                 accum_distance_weight: float,
                accum_enhanced_by_amount_travelled: bool,
                accum_exp_factor: float,
                normalize : bool = False,
                **kwargs ):
        """
        Initializes the reward manager with explicit reward weights and parameters
        for modeling centerline distance reward using a Gaussian function.

        Args:
            drive_forward_reward (float):   Weight for forward-acceleration acitons
    """
        
        super().__init__(normalize)

        self.terms = [AccumulatedDistanceReward(
                accum_distance_weight,
                enhanced_by_amount_travelled=accum_enhanced_by_amount_travelled,
                exponential_factor=accum_exp_factor,
            ),]
        if len(kwargs) > 0:
            print(f"Got additional kwargsuments that are not used; ignoring them : {kwargs.keys()}. Maybe they're used by a class that inherits.")
