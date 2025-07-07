import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class LateralDistanceDataset(Dataset):
    """Data-loader for dataset created with PretrainingDataCollection (@see trackmania_env.envs.testcases_single_agent.py)"""
    def __init__(self, root_dir, transform=None):
        """
        Parameters:
            - root_dir (str): Directory with 'images/' and 'labels.csv'
            - transform (callable, optional): Optional transform to apply to the images
        """
        self.img_dir = os.path.join(root_dir, "images")
        self.labels_df = pd.read_csv(os.path.join(root_dir, "labels.csv"))
        self.transform = transform


    def get_samples_by_indices(self, indices : list[int]) -> list[tuple[torch.Tensor,torch.Tensor]]:
        """Get samples by indices"""
        samples = []
        for idx in indices:
            img, label = self[idx]  # Leverage __getitem__
            samples.append((img, label))
        return samples

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_filename = self.labels_df.iloc[idx]["filename"]
        img_path = os.path.join(self.img_dir, img_filename)
        img = np.load(img_path) 

        img = torch.tensor(img, dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        # Load label
        label = self.labels_df.iloc[idx]["lateral_distance"]
        label = torch.tensor(label, dtype=torch.float32)

        return img, label
