
from gymnasium import spaces
import numpy as np

def get_flattened_dict_dim(space_list : list[spaces.Space]) -> int:
    total_dim = 0

    for space in space_list:
        if isinstance(space, spaces.Box):
            total_dim += int(np.prod(space.shape))

        elif isinstance(space, spaces.Discrete):
            total_dim += 1  # or `space.n` for one-hot

        elif isinstance(space, spaces.MultiBinary):
            total_dim += space.n

        elif isinstance(space, spaces.MultiDiscrete):
            total_dim += len(space.nvec)

        else:
            raise NotImplementedError(f"Unsupported space type ': {type(space)}'")

    return total_dim