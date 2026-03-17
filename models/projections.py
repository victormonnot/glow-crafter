"""
projections.py — Couches de projection domaine ↔ workspace
============================================================
C'est le PONT entre l'espace latent d'un domaine et l'espace partagé.

  DomainProjection  : z_d (latent_dim) → w_d (workspace_dim)
  InverseProjection : w   (workspace_dim) → ẑ_d (latent_dim)

Dans Shimmer, ils appellent ça GWEncoder/GWDecoder.
Noms pourris. Ici c'est clair : projection et projection inverse.

Architecture : MLP avec skip connections optionnelles.
"""

import torch
import torch.nn as nn


class DomainProjection(nn.Module):
    """Projette du latent domaine vers l'espace workspace partagé.
    z_d → w_d
    """

    def __init__(
        self,
        latent_dim: int,
        workspace_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, workspace_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → w: (B, workspace_dim)"""
        return self.net(z)


class InverseProjection(nn.Module):
    """Projette du workspace vers le latent domaine.
    w → ẑ_d

    Miroir de DomainProjection. Même archi, direction inverse.
    """

    def __init__(
        self,
        workspace_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_dim = workspace_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """w: (B, workspace_dim) → z_hat: (B, latent_dim)"""
        return self.net(w)
