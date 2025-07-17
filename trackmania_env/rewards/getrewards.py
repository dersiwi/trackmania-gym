from tminterface.structs import CheckpointData, SimStateData, CheckpointTime

from trackmania_env.utils.position_buffer import PositionBuffer
from trackmania_env.rewards.reward_calculation import RewradCalculator
from trackmania_env.rewards.basic_rewards import BasicRewardCalculation
from trackmania_env.rewards.linesight_rewards import LinesightRewardCalculator
from trackmania_env.rewards.nextpointrewards import NextPointRewards,NextPointRewards2, RaceFinishedRewards
from trackmania_env.rewards.sophy_rewards import SophyRewards
from configs.config import TrainConfig



def get_reward_calculator(reward_calculator_cfg: TrainConfig) -> RewradCalculator:
    name = reward_calculator_cfg.name

    match name:
        case "basic":
            return BasicRewardCalculation(reward_calculator_cfg)
        case "linesight":
            return LinesightRewardCalculator(reward_calculator_cfg)
        case "nextpoint":
            return NextPointRewards(reward_calculator_cfg)
        case "nextpoint2":
            return NextPointRewards2(reward_calculator_cfg)
        case "sophy":
            return SophyRewards(reward_calculator_cfg)
        case "race_finished":
            return RaceFinishedRewards(reward_calculator_cfg)
        case _:
            raise NameError(f"Reward calculator '{name}' not known.")