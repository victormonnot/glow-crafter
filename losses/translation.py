"""
translation.py — Loss de traduction cross-modale
==================================================
Pipeline : x_a → encode_a → z_a → proj_a → w_a → inv_proj_b → z_b → decode_b → x_b_hat
Loss    : distance(x_b_hat, x_b)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional


class TranslationLoss(nn.Module):
    """Loss de traduction pour signaux continus (vision, state). MSE par defaut."""

    def __init__(self, loss_fn: Optional[Callable] = None):
        super().__init__()
        self.loss_fn = loss_fn or F.mse_loss

    def forward(
        self,
        x_target: torch.Tensor,
        x_reconstructed: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss_fn(x_reconstructed, x_target)


class ActionTranslationLoss(nn.Module):
    """Variante pour les actions : cross-entropy sur les logits."""

    def forward(
        self,
        target_actions: torch.Tensor,
        predicted_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            target_actions:   (B,) — action indices ground truth
            predicted_logits: (B, num_actions) — logits predits
        """
        return F.cross_entropy(predicted_logits, target_actions)
