
from trackmania_env.rewards.reward_calculation import RewradCalculator
from trackmania_env.rewards.basic_rewards import BasicRewardCalculation
from trackmania_env.rewards.linesight_rewards import LinesightRewardCalculator
from trackmania_env.rewards.nextpointrewards import NextPointRewards,NextPointRewards2,NextPointRewards3 ,RaceFinishedRewards
from trackmania_env.rewards.sophy_rewards import SophyRewards
from configs.config import RewardManagerCfg



def get_reward_calculator(reward_calculator_cfg: RewardManagerCfg, normalize: bool = False) -> RewradCalculator:
    name = reward_calculator_cfg.name

    match name:
        case "basic":
            return BasicRewardCalculation(normalize)
        case "linesight":
            return LinesightRewardCalculator(**reward_calculator_cfg.args)
        case "nextpoint":
            return NextPointRewards(**reward_calculator_cfg.args, normalize=normalize)
        case "nextpoint2":
            return NextPointRewards2(**reward_calculator_cfg.args, normalize=normalize)
        case "nextpoint3":
            return NextPointRewards3(**reward_calculator_cfg.args, normalize=normalize)
        case "sophy":
            return SophyRewards(**reward_calculator_cfg.args, normalize=normalize)
        case "race_finished":
            return RaceFinishedRewards(**reward_calculator_cfg.args, 
                                       use_punishment=reward_calculator_cfg.use_punishment, 
                                       steps_without_progress_until_punishment=reward_calculator_cfg.steps_without_progress_until_punishment,
                                       normalize=normalize)
        case _:
            raise NameError(f"Reward calculator '{name}' not known.")