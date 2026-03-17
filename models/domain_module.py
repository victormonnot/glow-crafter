"""
domain_module.py — Le module spécialiste d'un domaine
======================================================
Regroupe les 4 composants d'un domaine en un seul objet :

  encoder          : x_d → z_d      (entrée brute → latent)
  decoder          : z_d → x̂_d      (latent → reconstruction)
  projection       : z_d → w_d      (latent → workspace)
  inv_projection   : w   → ẑ_d      (workspace → latent)

C'est l'unité atomique de GLoW. Un domaine = un DomainModule.
Ajouter une modalité = instancier un nouveau DomainModule.

Dans Shimmer c'est explosé en 3 niveaux d'héritage.
Ici c'est un objet. Point.
"""

import torch
import torch.nn as nn

from models.projections import DomainProjection, InverseProjection


class DomainModule(nn.Module):
    """Un domaine complet : encode, projette, déprojette, décode."""

    def __init__(
        self,
        name: str,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        workspace_dim: int,
        proj_hidden_dim: int = 256,
        proj_num_layers: int = 2,
        proj_dropout: float = 0.1,
    ):
        super().__init__()
        self.name = name
        self.latent_dim = latent_dim

        # Les 4 briques
        self.encoder = encoder
        self.decoder = decoder
        self.projection = DomainProjection(
            latent_dim, workspace_dim, proj_hidden_dim, proj_num_layers, proj_dropout
        )
        self.inv_projection = InverseProjection(
            workspace_dim, latent_dim, proj_hidden_dim, proj_num_layers, proj_dropout
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Entrée brute → latent domaine."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Latent domaine → reconstruction."""
        return self.decoder(z)

    def to_workspace(self, z: torch.Tensor) -> torch.Tensor:
        """Latent domaine → espace workspace."""
        return self.projection(z)

    def from_workspace(self, w: torch.Tensor) -> torch.Tensor:
        """Espace workspace → latent domaine."""
        return self.inv_projection(w)

    # --- Raccourcis utiles ---

    def encode_to_workspace(self, x: torch.Tensor) -> torch.Tensor:
        """Entrée brute → workspace (encode + projette)."""
        return self.to_workspace(self.encode(x))

    def workspace_to_reconstruction(self, w: torch.Tensor) -> torch.Tensor:
        """Workspace → reconstruction (déprojette + décode)."""
        return self.decode(self.from_workspace(w))

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Autoencoder classique : x → z → x̂ (sans passer par le workspace)."""
        return self.decode(self.encode(x))
