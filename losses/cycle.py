"""
cycle.py — Loss de cycle-consistency
======================================
Garantit que les allers-retours entre domaines via le workspace
ramènent au point de départ.

Cycle : z_a → w_a → ẑ_b → w_b → ẑ_a
Loss  : ||ẑ_a - z_a||

Pourquoi c'est important :
  - Contrastive aligne les domaines globalement (proche/loin)
  - Translation force la reconstruction cross-modale
  - Cycle force la COHÉRENCE : si A→B→A ≠ A, l'alignement est bancal

C'est particulièrement critique quand tu as >2 domaines, parce que
les paires contrastives ne garantissent pas la transitivité.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.domain_module import DomainModule


class CycleConsistencyLoss(nn.Module):
    """L2 loss sur le round-trip dans l'espace latent via le workspace."""

    def forward(
        self,
        z_source: torch.Tensor,
        z_reconstructed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_source:        (B, latent_dim) — latent original du domaine A
            z_reconstructed: (B, latent_dim) — latent après cycle A→B→A

        Returns:
            loss scalaire
        """
        return F.mse_loss(z_reconstructed, z_source)


def compute_cycle(
    domain_a: DomainModule,
    domain_b: DomainModule,
    z_a: torch.Tensor,
) -> torch.Tensor:
    """Effectue un cycle complet A → workspace → B → workspace → A.

    Args:
        domain_a: Module du domaine source
        domain_b: Module du domaine intermédiaire
        z_a: (B, latent_dim_a) — latent de départ

    Returns:
        z_a_reconstructed: (B, latent_dim_a) — après le round-trip
    """
    # A → workspace
    w_a = domain_a.to_workspace(z_a)
    # workspace → B (latent)
    z_b = domain_b.from_workspace(w_a)
    # B → workspace
    w_b = domain_b.to_workspace(z_b)
    # workspace → A (latent)
    z_a_hat = domain_a.from_workspace(w_b)

    return z_a_hat
