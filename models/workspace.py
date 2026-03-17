"""
workspace.py — Le Global Workspace
====================================
LE fichier central. C'est ici que les représentations de tous les
domaines se rencontrent, fusionnent, et repartent.

Inspiré de la Global Workspace Theory (Baars, 1988) :
  - Les modules spécialistes projettent dans l'espace partagé
  - Le workspace fusionne les représentations
  - Le résultat est "broadcasté" à tous les modules

Deux modes de fusion :
  "mean"      : Moyenne simple des w_d. Baseline.
  "attention" : Cross-attention entre les w_d. Plus expressif,
                permet au workspace de pondérer dynamiquement
                les contributions de chaque domaine.
"""

import torch
import torch.nn as nn
from typing import Dict

from models.domain_module import DomainModule


class GlobalWorkspace(nn.Module):
    """
    Le workspace global. Prend N domain modules, fusionne leurs
    représentations, et permet la traduction cross-modale.
    """

    def __init__(
        self,
        domain_modules: Dict[str, DomainModule],
        workspace_dim: int,
        fusion: str = "attention",
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.domain_modules = nn.ModuleDict(domain_modules)
        self.workspace_dim = workspace_dim
        self.fusion_type = fusion

        if fusion == "attention":
            # Cross-attention : chaque domaine "attend" aux autres
            layer = nn.TransformerEncoderLayer(
                d_model=workspace_dim,
                nhead=num_heads,
                dim_feedforward=workspace_dim * 4,
                batch_first=True,
            )
            self.fusion_net = nn.TransformerEncoder(layer, num_layers=num_layers)
        elif fusion == "mean":
            self.fusion_net = None
        else:
            raise ValueError(f"Fusion inconnue: {fusion}. Utilise 'mean' ou 'attention'.")

    def encode_domains(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Encode chaque domaine présent et projette dans le workspace.

        Args:
            inputs: {"vision": tensor, "text": tensor, ...}
                    Pas besoin que tous les domaines soient présents.

        Returns:
            {"vision": w_v, "text": w_t, ...} dans l'espace workspace.
        """
        workspace_reprs = {}
        for name, x in inputs.items():
            if name in self.domain_modules:
                workspace_reprs[name] = self.domain_modules[name].encode_to_workspace(x)
        return workspace_reprs

    def fuse(self, workspace_reprs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fusionne les représentations workspace de plusieurs domaines.

        Args:
            workspace_reprs: {"vision": w_v, "text": w_t, ...}
                            Chaque w_d est (B, workspace_dim)

        Returns:
            w_fused: (B, workspace_dim) — la représentation unifiée.
        """
        if len(workspace_reprs) == 1:
            # Un seul domaine, pas de fusion nécessaire
            return next(iter(workspace_reprs.values()))

        if self.fusion_type == "mean":
            # Moyenne simple
            stacked = torch.stack(list(workspace_reprs.values()), dim=0)  # (N, B, D)
            return stacked.mean(dim=0)

        elif self.fusion_type == "attention":
            # Empile les domaines comme une séquence pour le transformer
            # (B, N_domains, workspace_dim) — chaque domaine = un "token"
            stacked = torch.stack(list(workspace_reprs.values()), dim=1)
            fused = self.fusion_net(stacked)  # (B, N_domains, D)
            # Mean pool sur les domaines après attention
            return fused.mean(dim=1)  # (B, D)

    def translate(
        self,
        source_name: str,
        target_name: str,
        x_source: torch.Tensor,
    ) -> torch.Tensor:
        """Traduction cross-modale : domaine source → workspace → domaine cible.

        C'est la killer feature de GLoW.
        Ex: image → workspace → texte
        """
        source = self.domain_modules[source_name]
        target = self.domain_modules[target_name]

        z_source = source.encode(x_source)
        w = source.to_workspace(z_source)
        z_target = target.from_workspace(w)
        x_target = target.decode(z_target)
        return x_target

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Forward complet :
        1. Encode chaque domaine → workspace
        2. Fusionne
        3. Broadcast : décode vers tous les domaines

        Returns:
            reconstructions: {"vision": x̂_v, "text": x̂_t, ...}
        """
        # Encode + projette
        workspace_reprs = self.encode_domains(inputs)

        # Fusionne
        w_fused = self.fuse(workspace_reprs)

        # Broadcast : workspace → chaque domaine
        reconstructions = {}
        for name, module in self.domain_modules.items():
            reconstructions[name] = module.workspace_to_reconstruction(w_fused)

        return reconstructions, workspace_reprs, w_fused
