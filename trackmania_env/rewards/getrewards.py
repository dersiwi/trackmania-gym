from trackmania_env.rewards.reward_calculation import RewradCalculator
from configs.config import RewardManagerCfg

import hydra.utils

def get_reward_calculator_from_cfg(reward_calculator_cfg: RewardManagerCfg, normalize: bool = False) -> RewradCalculator:
    reward_calculator: RewradCalculator = hydra.utils.instantiate(
            reward_calculator_cfg,
            normalize = normalize,
            ) 

    return reward_calculator
