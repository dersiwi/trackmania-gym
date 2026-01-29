import os
from datetime import datetime
import wandb
import numpy as np
import torch
from gymnasium import spaces
from PIL import Image
from stable_baselines3.common.callbacks import BaseCallback

from trackmania_env.utils.return_tracker import ReturnTracker

class RewardLogCallback(BaseCallback):
    """
    This custom RewardLogCallback should log the rewards on a per-step basis and also log each reward-term individually.
    """
    def __init__(self, verbose=0):
        return super().__init__(verbose)

    def _on_step(self) -> bool:
        # have to call self.locals["infos"][0], because sb3 has an info-dict for each environment, since currently we only train with one environment, this index is always 0
        infos : list[dict] = self.locals["infos"][0]

        if "rewards" in infos and not len(infos["rewards"]) == 0:
            wandb.log(infos["rewards"])

        return True #always return true.
    

class AccumRewardLogCallback(BaseCallback):
    """
    This custom RewardLogCallback should log the individual, accumulated reward-terms after each episode ends.
    """
    def __init__(self, verbose=0, n_envs : int = 1):
        super().__init__(verbose)
        self.rewardterms_to_log : list[dict] = [{} for i in range(n_envs)]

    def _on_step(self) -> bool:
        # have to call self.locals["infos"][0], because sb3 has an info-dict for each environment, since currently we only train with one environment, this index is always 0
        #infos : list[dict] = self.locals["infos"][0]
        for env_idx in range(len(self.locals["infos"])):
            env_infos = self.locals["infos"][env_idx]

            if "rewards" in env_infos and not len(env_infos["rewards"]) == 0:

                for rewterm in env_infos["rewards"]:
                    if rewterm in self.rewardterms_to_log[env_idx]:
                        self.rewardterms_to_log[env_idx][rewterm + str(env_idx)] += env_infos["rewards"][rewterm]
                    else:
                        self.rewardterms_to_log[env_idx][rewterm + str(env_idx)] = env_infos["rewards"][rewterm]

            if ("terminated" in env_infos and env_infos["terminated"]) or ("truncated" in env_infos and env_infos["truncated"]):
                wandb.log(self.rewardterms_to_log[env_idx])
                self.rewardterms_to_log[env_idx] = {}

        return True #always return true.


class ReturnCallback(BaseCallback):
    """This callback listens to the environment wirting its episode-return into the infos and then logs this return per episode"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
    
    def _on_step(self):
        infos : list[dict] = self.locals["infos"][0]
        if ReturnTracker.LOG_NAME in infos:
            wandb.log({"episode_return" : infos[ReturnTracker.LOG_NAME]})
        return True # always return true.
    

class ContinuousActionLogCallback(BaseCallback):
    def __init__(self, verbose = 0, log_minmax = True):
        super().__init__(verbose)
        self.actionmin : dict[str, float] = {}
        self.actionmax : dict[str, float] = {}
        self.actionmean : dict[str, float] = {}
        self.minprefix = "min_action_dim:"
        self.maxprefix = "max_action_dim:"
        self.meanprefix = "mean_action_dim:"
        self.n_steps = 0
        self.log_minmax = log_minmax


    def _on_step(self):#info["action"]
        infos : list[dict] = self.locals["infos"][0]
        self.n_steps += 1
        if "action" in infos:
            if not type("action") == np.ndarray:
                pass
            action : np.ndarray = infos["action"]
            for dimidx in range(action.shape[0]):
                actionkey = str(dimidx)
                if actionkey in self.actionmin:
                    self.actionmin[self.minprefix + actionkey] = min(action[dimidx], self.actionmin[self.minprefix + actionkey])
                    self.actionmax[self.maxprefix + actionkey] = max(action[dimidx], self.actionmax[self.maxprefix + actionkey])
                    self.actionmean[self.meanprefix + actionkey] += action[dimidx] / self.n_steps #not reaaally true but close enough
                else:
                    self.actionmin[self.minprefix + actionkey] = action[dimidx]
                    self.actionmax[self.maxprefix + actionkey] = action[dimidx]
                    self.actionmean[self.meanprefix + actionkey] = action[dimidx]
        
        if ("terminated" in infos and infos["terminated"]) or ("truncated" in infos and infos["truncated"]):
            if self.log_minmax:
                wandb.log(self.actionmin)
                wandb.log(self.actionmax)
            wandb.log(self.actionmean)
            self.actionmin : dict[str, float] = {}
            self.actionmax : dict[str, float] = {}
            self.actionmean : dict[str, float] = {}

        return super()._on_step()


class ImageDumpCallback(BaseCallback):

    def __init__(
        self,
        verbose: int = 0,
        dump_freq: int = 1000000, # must be > 0 in order for images to be dumped
        dump_dir: str | None = None,
        dict_img_id: str | None = "image",
    ):
        super().__init__(verbose)

        self.dump_dir = dump_dir
        
        if dump_dir is None:
            dump_dir = f"./logs/image_dumps/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.dump_dir = os.path.abspath(dump_dir)
        os.makedirs(self.dump_dir, exist_ok=True)

        print(f"[ImageDumpCallback] Dumping images to: {self.dump_dir}")

        self.dump_freq = dump_freq
        self.dict_img_id = dict_img_id

        self.n_imgs = 0
        self.imgs_to_dump = 0

        self.is_dict_obs = False
        self.is_image_obs = False

    @staticmethod
    def is_image_space(space: spaces.Space) -> bool:
        """Checks whether a Gymnasium space can be interpreted as an image."""
        if not isinstance(space, spaces.Box):
            return False

        if space.shape is None:
            return False

        # Common image shapes:
        # (H, W), (1, H, W), (H, W, C), (C, H, W)
        if len(space.shape) in [2,3]:
            return True

        return False

    def save_image(self, img: np.ndarray | torch.Tensor, filepath: str) -> None:
        """Saves an image to the given path."""
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        if img.ndim == 4 : img = img.squeeze(0) # remove batch dim
        # Squeeze grayscale channel if needed
        if img.ndim == 3 and img.shape[0] == 1:
            img = img.squeeze(0)

        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(filepath)

    def set_dump_freq(self, freq: int, dirpath: str) -> None:
        self.dump_freq = freq
        self.dump_dir = dirpath
        assert freq > 5, "Cannot go lower than 5 due to magic numbers"

    def _on_training_start(self) -> None:

        obs_space = self.training_env.observation_space

        if isinstance(obs_space, spaces.Dict):
            self.is_dict_obs = True

            if self.dict_img_id not in obs_space.spaces:
                raise KeyError(
                    f"Dict observation space does not contain key '{self.dict_img_id}'. "
                    f"Available keys: {list(obs_space.spaces.keys())}"
                )

            img_space = obs_space.spaces[self.dict_img_id]

            if not self.is_image_space(img_space):
                raise TypeError(
                    f"Observation under key '{self.dict_img_id}' is not image-like. "
                    f"Shape: {img_space.shape}"
                )

            self.is_image_obs = True

        elif isinstance(obs_space, spaces.Box):
            self.is_dict_obs = False

            if not self.is_image_space(obs_space):
                raise TypeError(
                    f"Box observation space is not image-like. Shape: {obs_space.shape}"
                )

            self.is_image_obs = True

        else:
            raise TypeError(
                f"Unsupported observation space type: {type(obs_space)}. "
                "Only gymnasium.spaces.Box and spaces.Dict are supported."
            )

    def _on_step(self) -> bool:
        if not self.is_image_obs:
            return True

        obs = self.locals["obs_tensor"]

        if self.is_dict_obs:
            img = obs[self.dict_img_id]
        else:
            img = obs

        if self.imgs_to_dump > 0 or (self.dump_freq > 0 and self.n_imgs % self.dump_freq == 0):

            filepath = os.path.join(
                self.dump_dir, f"observation_img_{self.n_imgs}.png"
            )
            self.save_image(img, filepath)

            self.imgs_to_dump = (
                5 if self.dump_freq > 0 and self.n_imgs % self.dump_freq == 0 else self.imgs_to_dump
            )
            self.imgs_to_dump -= 1

        self.n_imgs += 1
        return True
