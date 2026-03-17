"""
eval.py — Évaluation du Global Workspace
==========================================
Deux métriques fondamentales :

1. Cross-modal retrieval : Est-ce que le workspace aligne bien ?
   Donne une image, cherche le texte le plus proche dans le workspace.
   Mesure Recall@K.

2. Translation quality : Est-ce que la traduction cross-modale marche ?
   image → workspace → texte reconstruit. Mesure la qualité.

Si ces métriques sont bonnes, ton workspace a appris une vraie
représentation partagée. Sinon, c'est du bruit organisé.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from tqdm import tqdm

from models.workspace import GlobalWorkspace


@torch.no_grad()
def compute_retrieval_metrics(
    workspace: GlobalWorkspace,
    dataloader: DataLoader,
    source_domain: str,
    target_domain: str,
    device: torch.device,
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """Cross-modal retrieval : Recall@K.

    Pour chaque sample, on projette le domaine source dans le workspace,
    et on cherche le plus proche voisin parmi tous les samples du
    domaine cible dans le workspace.

    Returns:
        {"R@1": 0.42, "R@5": 0.78, "R@10": 0.91}
    """
    workspace.eval()

    all_w_source = []
    all_w_target = []

    for batch in dataloader:
        x_source = batch[source_domain].to(device)
        x_target = batch[target_domain].to(device)

        w_source = workspace.domain_modules[source_domain].encode_to_workspace(x_source)
        w_target = workspace.domain_modules[target_domain].encode_to_workspace(x_target)

        all_w_source.append(F.normalize(w_source, dim=-1))
        all_w_target.append(F.normalize(w_target, dim=-1))

    # (N_total, workspace_dim)
    all_w_source = torch.cat(all_w_source, dim=0)
    all_w_target = torch.cat(all_w_target, dim=0)

    # Matrice de similarité (N, N)
    sim_matrix = torch.matmul(all_w_source, all_w_target.T)

    # Ground truth : la diagonale (sample i du source → sample i du target)
    N = sim_matrix.size(0)
    gt_indices = torch.arange(N, device=device)

    # Top-K indices
    _, topk_indices = sim_matrix.topk(max(k_values), dim=1)

    results = {}
    for k in k_values:
        # Pour chaque sample, est-ce que le bon match est dans le top-K ?
        correct = (topk_indices[:, :k] == gt_indices.unsqueeze(1)).any(dim=1)
        results[f"R@{k}"] = correct.float().mean().item()

    return results


@torch.no_grad()
def compute_translation_mse(
    workspace: GlobalWorkspace,
    dataloader: DataLoader,
    source_domain: str,
    target_domain: str,
    device: torch.device,
) -> float:
    """MSE de traduction cross-modale.

    source → workspace → target_reconstructed vs target_ground_truth
    Plus c'est bas, mieux c'est.
    """
    workspace.eval()
    total_mse = 0.0
    count = 0

    for batch in dataloader:
        x_source = batch[source_domain].to(device)
        x_target = batch[target_domain].to(device)

        x_target_hat = workspace.translate(source_domain, target_domain, x_source)
        total_mse += F.mse_loss(x_target_hat, x_target, reduction="sum").item()
        count += x_target.size(0)

    return total_mse / count


@torch.no_grad()
def compute_workspace_stats(
    workspace: GlobalWorkspace,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Stats de l'espace workspace pour debug.

    Vérifie que les représentations sont bien distribuées
    et que les domaines se chevauchent dans le workspace.
    """
    workspace.eval()

    domain_reprs = {name: [] for name in workspace.domain_modules}

    for batch in dataloader:
        reprs = workspace.encode_domains(
            {name: batch[name].to(device) for name in batch if name in workspace.domain_modules}
        )
        for name, w in reprs.items():
            domain_reprs[name].append(w)

    stats = {}
    means = {}

    for name, repr_list in domain_reprs.items():
        all_w = torch.cat(repr_list, dim=0)
        stats[f"{name}_mean_norm"] = all_w.norm(dim=-1).mean().item()
        stats[f"{name}_std"] = all_w.std().item()
        means[name] = all_w.mean(dim=0)

    # Distance entre les centroïdes des domaines
    names = list(means.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dist = (means[names[i]] - means[names[j]]).norm().item()
            cos = F.cosine_similarity(
                means[names[i]].unsqueeze(0), means[names[j]].unsqueeze(0)
            ).item()
            stats[f"centroid_dist_{names[i]}_{names[j]}"] = dist
            stats[f"centroid_cos_{names[i]}_{names[j]}"] = cos

    return stats


def evaluate(
    workspace: GlobalWorkspace,
    eval_loader: DataLoader,
    config: dict,
    device: torch.device,
) -> Dict[str, float]:
    """Évaluation complète."""
    results = {}
    domain_pairs = [tuple(p) for p in config["training"]["domain_pairs"]]

    print("=" * 60)
    print("ÉVALUATION")
    print("=" * 60)

    for source, target in domain_pairs:
        # Retrieval
        retrieval = compute_retrieval_metrics(
            workspace, eval_loader, source, target, device
        )
        for k, v in retrieval.items():
            key = f"retrieval_{source}→{target}_{k}"
            results[key] = v
            print(f"  {key}: {v:.4f}")

        # Translation MSE (skip pour text car c'est des tokens)
        if target != "text":
            mse = compute_translation_mse(
                workspace, eval_loader, source, target, device
            )
            key = f"translation_mse_{source}→{target}"
            results[key] = mse
            print(f"  {key}: {mse:.6f}")

    # Workspace stats
    stats = compute_workspace_stats(workspace, eval_loader, device)
    results.update(stats)
    print()
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    return results
