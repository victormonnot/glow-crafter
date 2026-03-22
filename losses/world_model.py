"""
world_model.py — Losses pour le world model (RSSM)
====================================================
4 composantes :
  1. KL divergence : posterior vs prior (regularise les latents stochastiques)
  2. Workspace reconstruction : predit w_t depuis (h_t, z_t)
  3. Reward prediction : predit r_t
  4. Continue prediction : predit si l'episode continue

Tricks de Dreamer v2 :
  - KL balancing : pondere asymetriquement la KL pour eviter le posterior collapse
  - Free nats : KL minimum avant penalisation (laisse de la liberte aux latents)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kl(
    post_mean: torch.Tensor,
    post_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
) -> torch.Tensor:
    """KL(posterior || prior) pour des gaussiennes diagonales."""
    var_ratio = (post_std / prior_std).pow(2)
    mean_diff = ((prior_mean - post_mean) / prior_std).pow(2)
    kl = 0.5 * (var_ratio + mean_diff - 1 - var_ratio.log())
    return kl.sum(dim=-1)  # somme sur stoch_dim, garde batch


class WorldModelLoss(nn.Module):
    """Loss combinee pour le world model."""

    def __init__(
        self,
        kl_weight: float = 1.0,
        kl_balance: float = 0.8,
        free_nats: float = 1.0,
        workspace_weight: float = 1.0,
        reward_weight: float = 1.0,
        continue_weight: float = 1.0,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.kl_balance = kl_balance
        self.free_nats = free_nats
        self.workspace_weight = workspace_weight
        self.reward_weight = reward_weight
        self.continue_weight = continue_weight

    def forward(
        self,
        post_mean: torch.Tensor,
        post_std: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_std: torch.Tensor,
        pred_workspace: torch.Tensor,
        target_workspace: torch.Tensor,
        pred_reward: torch.Tensor,
        target_reward: torch.Tensor,
        pred_continue: torch.Tensor,
        target_continue: torch.Tensor,
    ) -> dict:
        """
        Tous les tenseurs ont shape (B, L, ...) sauf targets qui peuvent varier.

        Returns:
            dict avec chaque loss + total
        """
        # --- KL avec balancing (Dreamer v2) ---
        # kl_balance > 0.5 : encourage le prior a suivre le posterior (pas l'inverse)
        # Evite que le posterior collapse vers le prior
        alpha = self.kl_balance

        # KL "forward" : train le prior (posterior detache)
        kl_prior = gaussian_kl(
            post_mean.detach(), post_std.detach(), prior_mean, prior_std
        )
        # KL "reverse" : train le posterior (prior detache)
        kl_post = gaussian_kl(
            post_mean, post_std, prior_mean.detach(), prior_std.detach()
        )

        # Free nats : ne penalise pas en dessous d'un seuil
        kl_prior = torch.clamp(kl_prior, min=self.free_nats).mean()
        kl_post = torch.clamp(kl_post, min=self.free_nats).mean()

        kl_loss = alpha * kl_prior + (1 - alpha) * kl_post

        # --- Workspace reconstruction ---
        workspace_loss = F.mse_loss(pred_workspace, target_workspace)

        # --- Reward prediction ---
        reward_loss = F.mse_loss(pred_reward, target_reward)

        # --- Continue prediction (binary) ---
        continue_loss = F.binary_cross_entropy_with_logits(
            pred_continue, target_continue
        )

        # --- Total ---
        total = (
            self.kl_weight * kl_loss
            + self.workspace_weight * workspace_loss
            + self.reward_weight * reward_loss
            + self.continue_weight * continue_loss
        )

        return {
            "total": total,
            "kl": kl_loss,
            "workspace_recon": workspace_loss,
            "reward": reward_loss,
            "continue": continue_loss,
        }
