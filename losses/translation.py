"""
translation.py — Loss de traduction cross-modale
==================================================
Vérifie qu'on peut reconstruire le domaine B à partir
de la représentation workspace du domaine A.

Pipeline : x_a → encode_a → z_a → proj_a → w_a → inv_proj_b → ẑ_b → decode_b → x̂_b
Loss    : distance(x̂_b, x_b)

C'est le test ultime de l'alignement : si le workspace aligne
vraiment les domaines, alors la représentation d'une image
devrait suffire à reconstruire le texte correspondant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional


class TranslationLoss(nn.Module):
    """Loss de traduction entre deux domaines via le workspace.

    Flexible : tu passes la fonction de reconstruction que tu veux.
    Par défaut : MSE pour les signaux continus, CE pour les tokens.
    """

    def __init__(self, loss_fn: Optional[Callable] = None):
        super().__init__()
        self.loss_fn = loss_fn or F.mse_loss

    def forward(
        self,
        x_target: torch.Tensor,
        x_reconstructed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_target:        (B, ...) — ground truth du domaine cible
            x_reconstructed: (B, ...) — reconstruction depuis le domaine source

        Returns:
            loss scalaire
        """
        return self.loss_fn(x_reconstructed, x_target)


class TextTranslationLoss(nn.Module):
    """Variante pour le texte : cross-entropy sur les logits."""

    def forward(
        self,
        target_tokens: torch.Tensor,
        predicted_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            target_tokens:   (B, seq_len) — token IDs ground truth
            predicted_logits: (B, seq_len, vocab_size) — logits prédits

        Returns:
            loss scalaire
        """
        B, S, V = predicted_logits.shape
        return F.cross_entropy(
            predicted_logits.reshape(B * S, V),
            target_tokens.reshape(B * S),
        )
