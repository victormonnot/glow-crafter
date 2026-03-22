"""
rssm.py — Recurrent State-Space Model dans l'espace workspace
===============================================================
Le coeur du world model. Dreamer-style RSSM qui opere dans le
workspace compact au lieu de l'espace pixel.

Etat RSSM = (h, z) :
  h = etat deterministe (GRU hidden), capture la structure temporelle
  z = etat stochastique, capture l'incertitude

Deux modes :
  - observe  : utilise l'observation reelle (posterior) — pour le training
  - imagine  : utilise seulement la dynamique (prior) — pour le planning

Le point cle : w_t (workspace fused) remplace l'observation pixel.
Le GW est gele, le RSSM apprend dans son espace compact.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class RSSM(nn.Module):
    """RSSM operant dans l'espace workspace."""

    def __init__(
        self,
        workspace_dim: int = 128,
        num_actions: int = 17,
        action_dim: int = 32,
        deter_dim: int = 256,
        stoch_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.workspace_dim = workspace_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.state_dim = deter_dim + stoch_dim  # for actor/critic input

        # Action embedding (discrete → continuous)
        self.action_embed = nn.Embedding(num_actions, action_dim)

        # Deterministic transition: h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
        self.gru_input = nn.Sequential(
            nn.Linear(stoch_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRUCell(hidden_dim, deter_dim)

        # Prior: p(z_t | h_t) — prediction sans observation
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),  # mean + log_std
        )

        # Posterior: q(z_t | h_t, w_t) — utilise l'observation workspace
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + workspace_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * stoch_dim),  # mean + log_std
        )

        # Predicteurs depuis l'etat RSSM (h, z)
        self.workspace_predictor = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, workspace_dim),
        )

        self.reward_predictor = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.continue_predictor = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def initial_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retourne l'etat initial (h_0, z_0) = zeros."""
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_dim, device=device)
        return h, z

    def _get_dist(self, stats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split stats en mean et std, clamp std pour stabilite."""
        mean, log_std = stats.chunk(2, dim=-1)
        std = F.softplus(log_std) + 0.1  # min std pour eviter le collapse
        return mean, std

    def _sample(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Reparameterized sampling."""
        return mean + std * torch.randn_like(std)

    def observe_step(
        self,
        prev_h: torch.Tensor,
        prev_z: torch.Tensor,
        prev_action: torch.Tensor,
        workspace_obs: torch.Tensor,
    ) -> dict:
        """Un step avec observation (training).

        Args:
            prev_h: (B, deter_dim)
            prev_z: (B, stoch_dim)
            prev_action: (B,) long
            workspace_obs: (B, workspace_dim) — w_t du GW gele

        Returns:
            dict avec h, z, prior_mean, prior_std, posterior_mean, posterior_std
        """
        # Transition deterministe
        a_emb = self.action_embed(prev_action)
        gru_in = self.gru_input(torch.cat([prev_z, a_emb], dim=-1))
        h = self.gru(gru_in, prev_h)

        # Prior
        prior_mean, prior_std = self._get_dist(self.prior_net(h))

        # Posterior (utilise l'observation)
        post_mean, post_std = self._get_dist(
            self.posterior_net(torch.cat([h, workspace_obs], dim=-1))
        )

        # Sample depuis posterior
        z = self._sample(post_mean, post_std)

        return {
            "h": h, "z": z,
            "prior_mean": prior_mean, "prior_std": prior_std,
            "post_mean": post_mean, "post_std": post_std,
        }

    def imagine_step(
        self,
        prev_h: torch.Tensor,
        prev_z: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Un step sans observation (imagination/planning).

        Returns:
            h: (B, deter_dim)
            z: (B, stoch_dim) — sample depuis prior
        """
        a_emb = self.action_embed(action)
        gru_in = self.gru_input(torch.cat([prev_z, a_emb], dim=-1))
        h = self.gru(gru_in, prev_h)

        prior_mean, prior_std = self._get_dist(self.prior_net(h))
        z = self._sample(prior_mean, prior_std)

        return h, z

    def observe_sequence(
        self,
        actions: torch.Tensor,
        workspace_obs_seq: torch.Tensor,
    ) -> dict:
        """Deroule observe_step sur une sequence.

        Args:
            actions: (B, L) long
            workspace_obs_seq: (B, L, workspace_dim)

        Returns:
            dict avec sequences de h, z, prior/posterior stats
        """
        B, L = actions.shape
        device = actions.device

        h, z = self.initial_state(B, device)

        hs, zs = [], []
        prior_means, prior_stds = [], []
        post_means, post_stds = [], []

        for t in range(L):
            out = self.observe_step(h, z, actions[:, t], workspace_obs_seq[:, t])
            h, z = out["h"], out["z"]

            hs.append(h)
            zs.append(z)
            prior_means.append(out["prior_mean"])
            prior_stds.append(out["prior_std"])
            post_means.append(out["post_mean"])
            post_stds.append(out["post_std"])

        return {
            "h": torch.stack(hs, dim=1),               # (B, L, deter_dim)
            "z": torch.stack(zs, dim=1),               # (B, L, stoch_dim)
            "prior_mean": torch.stack(prior_means, 1),  # (B, L, stoch_dim)
            "prior_std": torch.stack(prior_stds, 1),
            "post_mean": torch.stack(post_means, 1),
            "post_std": torch.stack(post_stds, 1),
        }

    def predict(self, h: torch.Tensor, z: torch.Tensor) -> dict:
        """Predit workspace, reward, continue depuis l'etat RSSM.

        Args:
            h: (..., deter_dim)
            z: (..., stoch_dim)

        Returns:
            dict avec pred_workspace, pred_reward, pred_continue
        """
        state = torch.cat([h, z], dim=-1)
        return {
            "workspace": self.workspace_predictor(state),
            "reward": self.reward_predictor(state).squeeze(-1),
            "continue": self.continue_predictor(state).squeeze(-1),
        }
