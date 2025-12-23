import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 16):
        """
        Encoder mapping 84x84 grayscale frames to a latent vector.
        """
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 1, 84, 84) normalized frames.
        Returns:
            L2-normalized latent vectors (batch_size, latent_dim).
        """
        features = self.conv(x)
        flattened = features.view(features.size(0), -1)
        latent_vector = self.fc(flattened)
        normalized_latent_vector = nn.functional.normalize(latent_vector, p=2, dim=1)
        return normalized_latent_vector


class InverseModel(nn.Module):
    def __init__(self, latent_dim: int = 16, num_actions: int = 5, hidden_dim: int = 32):
        """
        Predicts action logits from consecutive latent states.
        """
        super().__init__()
        self.input_dim = latent_dim * 2
        self.num_actions = num_actions
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_actions),
        )

    def forward(self, z_t: torch.Tensor, z_t1: torch.Tensor) -> torch.Tensor:
        combined_z = torch.cat((z_t, z_t1), dim=1)
        action_logits = self.network(combined_z)
        return action_logits

