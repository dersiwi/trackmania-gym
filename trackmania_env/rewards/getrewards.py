from tminterface.structs import CheckpointData, SimStateData, CheckpointTime

from trackmania_env.utils.position_buffer import PositionBuffer
from trackmania_env.rewards.reward_calculation import RewradCalculator
from trackmania_env.rewards.basic_rewards import BasicRewardCalculation
from trackmania_env.rewards.linesight_rewards import LinesightRewardCalculator
from trackmania_env.rewards.nextpointrewards import NextPointRewards
from configs.config import TrainConfig




def get_reward_calculator(cfg : TrainConfig) -> RewradCalculator:

    reward_calculator = cfg.rl_env.env.reward_calculator
    if reward_calculator == "basic":
        return BasicRewardCalculation(cfg.rl_env.reward_manager)
    elif reward_calculator == "linesight":
        return LinesightRewardCalculator(cfg.rl_env.reward_manager)
    elif reward_calculator == "nextpoint":
        return NextPointRewards(cfg.rl_env.reward_manager, cfg.gmi.reference_line)
    else:
        raise NameError(f"Rewardcalculator '{reward_calculator}' not known.")