"""
eval.py — Evaluation du Global Workspace (Crafter)
====================================================
1. Cross-modal retrieval (Recall@K)
2. Translation quality (MSE pour continu, accuracy pour action)
3. Workspace stats (norms, centroid distances)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional

from models.workspace import GlobalWorkspace
from models.rssm import RSSM
from models.actor_critic import Actor
from data.collector import extract_state


@torch.no_grad()
def compute_retrieval_metrics(
    workspace: GlobalWorkspace,
    dataloader: DataLoader,
    source_domain: str,
    target_domain: str,
    device: torch.device,
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """Cross-modal retrieval : Recall@K."""
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

    all_w_source = torch.cat(all_w_source, dim=0)
    all_w_target = torch.cat(all_w_target, dim=0)

    sim_matrix = torch.matmul(all_w_source, all_w_target.T)
    N = sim_matrix.size(0)
    gt_indices = torch.arange(N, device=device)

    _, topk_indices = sim_matrix.topk(max(k_values), dim=1)

    results = {}
    for k in k_values:
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
    """MSE de traduction cross-modale (pour domaines continus)."""
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
def compute_action_translation_accuracy(
    workspace: GlobalWorkspace,
    dataloader: DataLoader,
    source_domain: str,
    device: torch.device,
) -> float:
    """Accuracy de traduction vers le domaine action."""
    workspace.eval()
    correct = 0
    total = 0

    for batch in dataloader:
        x_source = batch[source_domain].to(device)
        actions_gt = batch["action"].to(device)

        logits = workspace.translate(source_domain, "action", x_source)
        preds = logits.argmax(dim=-1)
        correct += (preds == actions_gt).sum().item()
        total += actions_gt.size(0)

    return correct / total if total > 0 else 0.0


@torch.no_grad()
def compute_workspace_stats(
    workspace: GlobalWorkspace,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Stats de l'espace workspace pour debug."""
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
    """Evaluation complete."""
    results = {}
    domain_pairs = [tuple(p) for p in config["training"]["domain_pairs"]]

    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    for source, target in domain_pairs:
        # Retrieval
        retrieval = compute_retrieval_metrics(
            workspace, eval_loader, source, target, device
        )
        for k, v in retrieval.items():
            key = f"retrieval_{source}->{target}_{k}"
            results[key] = v
            print(f"  {key}: {v:.4f}")

        # Translation MSE
        mse = compute_translation_mse(
            workspace, eval_loader, source, target, device
        )
        key = f"translation_mse_{source}->{target}"
        results[key] = mse
        print(f"  {key}: {mse:.6f}")

    # Workspace stats
    stats = compute_workspace_stats(workspace, eval_loader, device)
    results.update(stats)
    print()
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    return results


@torch.no_grad()
def evaluate_crafter_agent(
    workspace: GlobalWorkspace,
    rssm: RSSM,
    actor: Actor,
    device: torch.device,
    num_episodes: int = 20,
    seed: int = 1000,
) -> Dict[str, float]:
    """Evalue l'agent entraine dans Crafter.

    Joue num_episodes, track reward total et longueur.
    """
    import crafter

    workspace.eval()
    rssm.eval()

    all_rewards = []
    all_lengths = []
    all_achievements = []

    for ep in range(num_episodes):
        env = crafter.Env(seed=seed + ep)
        obs = env.reset()
        done = False

        # Init RSSM state
        h, z = rssm.initial_state(1, device)
        prev_action = torch.zeros(1, dtype=torch.long, device=device)

        ep_reward = 0.0
        ep_length = 0
        info = None

        while not done:
            obs_t = torch.from_numpy(obs).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

            if info is None:
                state_t = torch.zeros(1, 16, device=device)
                state_t[0, :4] = 1.0  # health/food/drink/energy = 9/9 = 1.0
            else:
                state_t = torch.from_numpy(extract_state(info)).unsqueeze(0).to(device)

            w_t = workspace.encode_to_fused({"vision": obs_t, "state": state_t})

            out = rssm.observe_step(h, z, prev_action, w_t)
            h, z = out["h"], out["z"]

            action = actor.get_action(h, z, sample=False)
            action_int = action.item()

            obs, reward, done, info = env.step(action_int)
            prev_action = action

            ep_reward += reward
            ep_length += 1

        all_rewards.append(ep_reward)
        all_lengths.append(ep_length)
        all_achievements.append(info.get("achievements", {}))

    results = {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "mean_length": float(np.mean(all_lengths)),
        "max_reward": float(np.max(all_rewards)),
    }

    achievement_counts = {}
    for ep_ach in all_achievements:
        for k, v in ep_ach.items():
            if v > 0:
                achievement_counts[k] = achievement_counts.get(k, 0) + 1
    results["unique_achievements"] = len(achievement_counts)
    results["achievement_details"] = achievement_counts

    print()
    print("=" * 60)
    print("CRAFTER AGENT EVALUATION")
    print("=" * 60)
    print(f"  Mean reward:  {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"  Max reward:   {results['max_reward']:.2f}")
    print(f"  Mean length:  {results['mean_length']:.0f}")
    print(f"  Achievements: {results['unique_achievements']} unique types")
    if achievement_counts:
        for k, v in sorted(achievement_counts.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v}/{num_episodes} episodes")

    return results
