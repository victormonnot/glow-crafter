"""
actor_critic.py — Actor et Critic pour le planning en imagination
==================================================================
Operent dans l'espace latent RSSM (h, z), pas dans l'espace pixel.

Actor  : pi(a | h, z) — politique, choisit les actions
Critic : V(h, z) — fonction de valeur, evalue les etats

Pour le backprop a travers les actions discretes, on utilise
straight-through + entropy regularization (Dreamer v2 approach).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Actor(nn.Module):
    """Politique : etat RSSM → distribution sur les actions."""

    def __init__(self, state_dim: int, hidden_dim: int = 256, num_actions: int = 17):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> Categorical:
        """
        Args:
            h: (..., deter_dim)
            z: (..., stoch_dim)

        Returns:
            Distribution Categorical sur les actions
        """
        state = torch.cat([h, z], dim=-1)
        logits = self.net(state)
        return Categorical(logits=logits)

    def get_action(self, h: torch.Tensor, z: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Sample ou greedy action."""
        dist = self.forward(h, z)
        if sample:
            return dist.sample()
        return dist.probs.argmax(dim=-1)


class Critic(nn.Module):
    """Fonction de valeur : etat RSSM → valeur scalaire."""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (..., deter_dim)
            z: (..., stoch_dim)

        Returns:
            values: (...,) scalaire
        """
        state = torch.cat([h, z], dim=-1)
        return self.net(state).squeeze(-1)
