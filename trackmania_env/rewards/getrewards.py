from tminterface.structs import CheckpointData, SimStateData, CheckpointTime

from trackmania_env.utils.position_buffer import PositionBuffer
from trackmania_env.rewards.reward_calculation import RewradCalculator
from trackmania_env.rewards.basic_rewards import BasicRewardCalculation
from trackmania_env.rewards.linesight_rewards import LinesightRewardCalculator


from game_interaction.ipc_fields import IPCFields
import numpy as np



def get_reward_calculator(reward_calculator : str, position_buffer : PositionBuffer) -> RewradCalculator:

    if reward_calculator == "basic":
        return BasicRewardCalculation(position_buffer)
    elif reward_calculator == "linesight":
        return LinesightRewardCalculator(position_buffer)
    else:
        raise NameError(f"Rewardcalculator '{reward_calculator}' not known.")