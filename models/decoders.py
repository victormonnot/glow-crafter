"""
decoders.py — Decodeurs domaine-specifiques (Crafter)
======================================================
Miroir des encodeurs. Reconstruit l'entree brute depuis le latent.
"""

import torch
import torch.nn as nn


class VisionDecoder(nn.Module):
    """z_vision → Image 64x64 reconstruite."""

    def __init__(self, latent_dim: int = 256, hidden_dim: int = 64, output_channels: int = 3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim * 4 * 4 * 4)
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            # 4x4 → 8x8
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(),
            # 8x8 → 16x16
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            # 16x16 → 32x32
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            # 32x32 → 64x64
            nn.ConvTranspose2d(hidden_dim, output_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → x_hat: (B, 3, 64, 64)"""
        h = self.fc(z).view(-1, self.hidden_dim * 4, 4, 4)
        return self.net(h)


class StateDecoder(nn.Module):
    """z_state → State vector reconstruit."""

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 64, output_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),  # State normalise [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → state_hat: (B, 16)"""
        return self.net(z)


class ActionDecoder(nn.Module):
    """z_action → Action logits."""

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 64, num_actions: int = 17):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → logits: (B, 17)"""
        return self.net(z)
