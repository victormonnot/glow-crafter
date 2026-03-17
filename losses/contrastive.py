"""
contrastive.py — Loss contrastive dans l'espace workspace
===========================================================
Force les représentations workspace d'un même sample (provenant
de domaines différents) à être proches, et celles de samples
différents à être éloignées.

C'est exactement le même principe que CLIP, mais appliqué dans
l'espace workspace entre N'IMPORTE quelle paire de domaines.

InfoNCE : -log( exp(sim(w_i, w_j)/τ) / Σ_k exp(sim(w_i, w_k)/τ) )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """InfoNCE symétrique entre deux ensembles de représentations workspace."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, w_a: torch.Tensor, w_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            w_a: (B, D) — workspace repr du domaine A
            w_b: (B, D) — workspace repr du domaine B
            Les lignes i de w_a et w_b viennent du même sample (paires positives).

        Returns:
            loss scalaire
        """
        # Normalise (cosine similarity)
        w_a = F.normalize(w_a, dim=-1)
        w_b = F.normalize(w_b, dim=-1)

        # Matrice de similarité (B, B)
        logits = torch.matmul(w_a, w_b.T) / self.temperature

        # Les positives sont sur la diagonale
        labels = torch.arange(logits.size(0), device=logits.device)

        # Symétrique : A→B et B→A
        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.T, labels)

        return (loss_ab + loss_ba) / 2
