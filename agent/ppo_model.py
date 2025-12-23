import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class IceSliderCNN(BaseFeaturesExtractor):
    """Nature CNN encoder for IceSlider observations."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Normalize pixel values
        x = observations / 255.0
        x = self.cnn(x)
        return self.linear(x)


def make_policy_kwargs(features_dim: int = 512):
    """Return policy kwargs for PPO with the custom CNN and MLP heads."""
    return {
        "features_extractor_class": IceSliderCNN,
        "features_extractor_kwargs": {"features_dim": features_dim},
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
    }


