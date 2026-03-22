"""
encoders.py — Encodeurs domaine-specifiques (Crafter)
======================================================
Trois domaines :
  - Vision : 64x64 RGB frames → CNN
  - State  : 16 scalars (vitals + inventory) → MLP
  - Action : discrete action index → Embedding + MLP
"""

import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """Image 64x64 → z_vision. CNN 4 couches."""

    def __init__(self, input_channels: int = 3, hidden_dim: int = 64, latent_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            # 64x64 → 32x32
            nn.Conv2d(input_channels, hidden_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            # 32x32 → 16x16
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            # 16x16 → 8x8
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(),
            # 8x8 → 4x4
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim * 4, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 64, 64) → z: (B, latent_dim)"""
        return self.net(x)


class StateEncoder(nn.Module):
    """State vector (vitals + inventory) → z_state. MLP."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 16) → z: (B, latent_dim)"""
        return self.net(x)


class ActionEncoder(nn.Module):
    """Discrete action → z_action. Embedding + MLP."""

    def __init__(self, num_actions: int = 17, embed_dim: int = 32, hidden_dim: int = 64, latent_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(num_actions, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,) long → z: (B, latent_dim)"""
        return self.mlp(self.embedding(x))
