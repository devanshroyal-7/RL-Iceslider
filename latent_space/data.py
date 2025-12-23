"""
Dataset and dataloader for IceSlider latent-space training.
Loads (s_t, s_t1, a_t) tuples of preprocessed 84x84 grayscale frames.
"""

import pickle
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class IceSliderExperienceDataset(Dataset):
    """
    PyTorch Dataset for IceSlider grayscale (s_t, s_t+1, a_t) tuples.
    """

    def __init__(self, experience_path: str):
        """
        Args:
            experience_path: Path to pickled list of (state, next_state, action) tuples.
                             Each state is an 84x84 numpy array (grayscale).
        """
        print(f"Loading experience from {experience_path}...")
        with open(experience_path, "rb") as f:
            self.experience = pickle.load(f)

        print(f"Loaded {len(self.experience)} samples.")

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),  # (H, W) -> (1, H, W), scales to [0,1]
                transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1]
            ]
        )

    def __len__(self) -> int:
        return len(self.experience)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_t, state_t1, action_t = self.experience[idx]

        state_t = self.transform(state_t)
        state_t1 = self.transform(state_t1)
        action_t = torch.tensor(action_t, dtype=torch.long)

        return state_t, state_t1, action_t


def create_dataloader(
    experience_path: str, batch_size: int, shuffle: bool = True, num_workers: int = 4
) -> DataLoader:
    dataset = IceSliderExperienceDataset(experience_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

