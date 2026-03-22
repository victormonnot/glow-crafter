"""
train.py — Pipeline d'entrainement GLoW Crafter
=================================================
Phase 1 (Pretrain) : Autoencoders domaine par domaine
Phase 2 (Align)    : Alignement workspace (contrastive + translation + cycle)
Phase 3 (World Model) : RSSM dans l'espace workspace (workspace gele)
Phase 4 (Actor-Critic) : Policy + value en imagination

Chaque phase sauvegarde un checkpoint independant.
On peut relancer depuis n'importe quelle phase avec --phase N.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional

from models.workspace import GlobalWorkspace
from models.rssm import RSSM
from models.actor_critic import Actor, Critic
from losses.contrastive import ContrastiveLoss
from losses.translation import TranslationLoss
from losses.cycle import CycleConsistencyLoss, compute_cycle
from losses.world_model import WorldModelLoss
from pipeline.imagine import imagine_trajectories, compute_lambda_returns


def pretrain_epoch(
    workspace: GlobalWorkspace,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Phase 1 : Entraine chaque domaine comme autoencoder."""
    workspace.train()
    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device)

        for name, module in workspace.domain_modules.items():
            if name not in batch:
                continue

            x = batch[name].to(device)
            x_hat = module.reconstruct(x)
            loss += F.mse_loss(x_hat, x)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def align_epoch(
    workspace: GlobalWorkspace,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    domain_pairs: List[Tuple[str, str]],
    loss_weights: Dict[str, float],
    contrastive_loss_fn: ContrastiveLoss,
    translation_loss_fn: TranslationLoss,
    cycle_loss_fn: CycleConsistencyLoss,
) -> Dict[str, float]:
    """Phase 2 : Aligne les domaines dans le workspace."""
    workspace.train()
    metrics = {"contrastive": 0.0, "translation": 0.0, "cycle": 0.0, "total": 0.0}

    for batch in dataloader:
        optimizer.zero_grad()

        inputs = {name: batch[name].to(device) for name in batch if name in workspace.domain_modules}
        workspace_reprs = workspace.encode_domains(inputs)

        latents = {}
        for name in inputs:
            latents[name] = workspace.domain_modules[name].encode(inputs[name])

        loss_contrastive = torch.tensor(0.0, device=device)
        loss_translation = torch.tensor(0.0, device=device)
        loss_cycle = torch.tensor(0.0, device=device)

        for (name_a, name_b) in domain_pairs:
            if name_a not in workspace_reprs or name_b not in workspace_reprs:
                continue

            w_a = workspace_reprs[name_a]
            w_b = workspace_reprs[name_b]
            mod_a = workspace.domain_modules[name_a]
            mod_b = workspace.domain_modules[name_b]

            loss_contrastive += contrastive_loss_fn(w_a, w_b)

            z_b_hat = mod_b.from_workspace(w_a)
            x_b_hat = mod_b.decode(z_b_hat)
            loss_translation += translation_loss_fn(inputs[name_b], x_b_hat)

            z_a_hat = mod_a.from_workspace(w_b)
            x_a_hat = mod_a.decode(z_a_hat)
            loss_translation += translation_loss_fn(inputs[name_a], x_a_hat)

            z_a_cycle = compute_cycle(mod_a, mod_b, latents[name_a])
            loss_cycle += cycle_loss_fn(latents[name_a].detach(), z_a_cycle)

            z_b_cycle = compute_cycle(mod_b, mod_a, latents[name_b])
            loss_cycle += cycle_loss_fn(latents[name_b].detach(), z_b_cycle)

        total_loss = (
            loss_weights["contrastive"] * loss_contrastive
            + loss_weights["translation"] * loss_translation
            + loss_weights["cycle"] * loss_cycle
        )

        total_loss.backward()
        nn.utils.clip_grad_norm_(workspace.parameters(), max_norm=1.0)
        optimizer.step()

        metrics["contrastive"] += loss_contrastive.item()
        metrics["translation"] += loss_translation.item()
        metrics["cycle"] += loss_cycle.item()
        metrics["total"] += total_loss.item()

    n = len(dataloader)
    return {k: v / n for k, v in metrics.items()}


# ==================== PHASE 3 : World Model ====================

def worldmodel_epoch(
    workspace: GlobalWorkspace,
    rssm: RSSM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    wm_loss_fn: WorldModelLoss,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Phase 3 : Entraine le RSSM dans l'espace workspace gele."""
    workspace.eval()
    rssm.train()

    metrics = {"total": 0.0, "kl": 0.0, "workspace_recon": 0.0, "reward": 0.0, "continue": 0.0}
    n_batches = 0

    for batch in dataloader:
        if max_batches is not None and n_batches >= max_batches:
            break

        optimizer.zero_grad()

        vision = batch["vision"].to(device)
        state = batch["state"].to(device)
        action = batch["action"].to(device)
        reward = batch["reward"].to(device)
        done = batch["done"].to(device)

        B, L = action.shape

        with torch.no_grad():
            w_seq = []
            for t in range(L):
                inputs_t = {"vision": vision[:, t], "state": state[:, t]}
                w_t = workspace.encode_to_fused(inputs_t)
                w_seq.append(w_t)
            w_seq = torch.stack(w_seq, dim=1)

        rssm_out = rssm.observe_sequence(action, w_seq)
        preds = rssm.predict(rssm_out["h"], rssm_out["z"])

        losses = wm_loss_fn(
            post_mean=rssm_out["post_mean"],
            post_std=rssm_out["post_std"],
            prior_mean=rssm_out["prior_mean"],
            prior_std=rssm_out["prior_std"],
            pred_workspace=preds["workspace"],
            target_workspace=w_seq,
            pred_reward=preds["reward"],
            target_reward=reward,
            pred_continue=preds["continue"],
            target_continue=1.0 - done,
        )

        losses["total"].backward()
        nn.utils.clip_grad_norm_(rssm.parameters(), max_norm=1.0)
        optimizer.step()

        for k in metrics:
            metrics[k] += losses[k].item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in metrics.items()}


# ==================== PHASE 4 : Actor-Critic ====================

def actor_critic_epoch(
    workspace: GlobalWorkspace,
    rssm: RSSM,
    actor: Actor,
    critic: Critic,
    dataloader: DataLoader,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    device: torch.device,
    horizon: int = 15,
    gamma: float = 0.997,
    lambda_: float = 0.95,
    entropy_weight: float = 3e-4,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Phase 4 : Entraine actor et critic sur des trajectoires imaginees."""
    workspace.eval()
    rssm.eval()
    actor.train()
    critic.train()

    metrics = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "mean_reward": 0.0}
    n_batches = 0

    for batch in dataloader:
        if max_batches is not None and n_batches >= max_batches:
            break

        vision = batch["vision"].to(device)
        state = batch["state"].to(device)
        action = batch["action"].to(device)

        B, L = action.shape

        with torch.no_grad():
            w_seq = []
            for t in range(L):
                inputs_t = {"vision": vision[:, t], "state": state[:, t]}
                w_t = workspace.encode_to_fused(inputs_t)
                w_seq.append(w_t)
            w_seq = torch.stack(w_seq, dim=1)

            rssm_out = rssm.observe_sequence(action, w_seq)

            t_start = L // 2
            init_h = rssm_out["h"][:, t_start]
            init_z = rssm_out["z"][:, t_start]

        imagined = imagine_trajectories(rssm, actor, init_h, init_z, horizon)

        with torch.no_grad():
            values_target = critic(imagined["h"], imagined["z"])

        returns = compute_lambda_returns(
            imagined["reward"], values_target, imagined["continue"],
            gamma=gamma, lambda_=lambda_,
        )

        # Critic
        critic_optimizer.zero_grad()
        values = critic(imagined["h"].detach(), imagined["z"].detach())
        critic_loss = F.mse_loss(values, returns.detach())
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
        critic_optimizer.step()

        # Actor
        actor_optimizer.zero_grad()
        dists = actor(imagined["h"].detach(), imagined["z"].detach())
        log_probs = dists.log_prob(imagined["action"].detach())
        entropy = dists.entropy().mean()
        actor_loss = -(log_probs * returns.detach()).mean() - entropy_weight * entropy
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        actor_optimizer.step()

        metrics["actor_loss"] += actor_loss.item()
        metrics["critic_loss"] += critic_loss.item()
        metrics["entropy"] += entropy.item()
        metrics["mean_reward"] += imagined["reward"].mean().item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in metrics.items()}


# ==================== FONCTIONS PAR PHASE ====================

def train_phase1(workspace, train_loader, config, device):
    """Phase 1 : Pretrain autoencoders."""
    tc = config["training"]

    print("=" * 60)
    print("PHASE 1 — Pretrain autoencoders")
    print("=" * 60)

    optimizer = torch.optim.AdamW(
        workspace.parameters(), lr=tc["pretrain_lr"], weight_decay=tc["weight_decay"]
    )

    last_loss = 0.0
    for epoch in range(tc["pretrain_epochs"]):
        last_loss = pretrain_epoch(workspace, train_loader, optimizer, device)
        print(f"  Epoch {epoch+1:3d}/{tc['pretrain_epochs']} — recon loss: {last_loss:.4f}")

    return {"recon_loss": last_loss}


def train_phase2(workspace, train_loader, config, device):
    """Phase 2 : Workspace alignment."""
    tc = config["training"]
    lc = config["losses"]

    print("=" * 60)
    print("PHASE 2 — Workspace alignment")
    print("=" * 60)

    contrastive_fn = ContrastiveLoss(temperature=lc["contrastive"]["temperature"])
    translation_fn = TranslationLoss()
    cycle_fn = CycleConsistencyLoss()

    loss_weights = {
        "contrastive": lc["contrastive"]["weight"],
        "translation": lc["translation"]["weight"],
        "cycle": lc["cycle"]["weight"],
    }
    domain_pairs = [tuple(p) for p in tc["domain_pairs"]]

    optimizer = torch.optim.AdamW(
        workspace.parameters(), lr=tc["align_lr"], weight_decay=tc["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tc["align_epochs"]
    )

    for epoch in range(tc["align_epochs"]):
        metrics = align_epoch(
            workspace=workspace,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            domain_pairs=domain_pairs,
            loss_weights=loss_weights,
            contrastive_loss_fn=contrastive_fn,
            translation_loss_fn=translation_fn,
            cycle_loss_fn=cycle_fn,
        )
        scheduler.step()

        print(
            f"  Epoch {epoch+1:3d}/{tc['align_epochs']} — "
            f"total: {metrics['total']:.4f}  "
            f"contr: {metrics['contrastive']:.4f}  "
            f"trans: {metrics['translation']:.4f}  "
            f"cycle: {metrics['cycle']:.4f}"
        )

    return metrics


def train_phase3(workspace, rssm, seq_loader, config, device):
    """Phase 3 : World Model (RSSM)."""
    tc = config["training"]
    lc = config["losses"]

    print("=" * 60)
    print("PHASE 3 — World Model (RSSM)")
    print("=" * 60)

    wm_lc = lc.get("world_model", {})
    wm_loss_fn = WorldModelLoss(
        kl_weight=wm_lc.get("kl_weight", 1.0),
        kl_balance=wm_lc.get("kl_balance", 0.8),
        free_nats=wm_lc.get("free_nats", 1.0),
        workspace_weight=wm_lc.get("workspace_recon_weight", 1.0),
        reward_weight=wm_lc.get("reward_weight", 1.0),
        continue_weight=wm_lc.get("continue_weight", 1.0),
    )

    wm_optimizer = torch.optim.AdamW(
        rssm.parameters(), lr=tc.get("wm_lr", 3e-4), weight_decay=tc["weight_decay"]
    )

    max_batches = tc.get("wm_max_batches", None)
    n_epochs = tc.get("wm_epochs", 100)

    for epoch in range(n_epochs):
        metrics = worldmodel_epoch(
            workspace, rssm, seq_loader, wm_optimizer, wm_loss_fn, device,
            max_batches=max_batches,
        )
        print(
            f"  Epoch {epoch+1:3d}/{n_epochs} — "
            f"total: {metrics['total']:.4f}  "
            f"kl: {metrics['kl']:.4f}  "
            f"ws_recon: {metrics['workspace_recon']:.4f}  "
            f"reward: {metrics['reward']:.4f}  "
            f"cont: {metrics['continue']:.4f}"
        )

    return metrics


def train_phase4(workspace, rssm, actor, critic, seq_loader, config, device):
    """Phase 4 : Actor-Critic (imagination)."""
    tc = config["training"]
    ac_cfg = config.get("actor_critic", {})

    print("=" * 60)
    print("PHASE 4 — Actor-Critic (imagination)")
    print("=" * 60)

    actor_optimizer = torch.optim.AdamW(
        actor.parameters(), lr=ac_cfg.get("actor_lr", 3e-5)
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(), lr=ac_cfg.get("critic_lr", 3e-5)
    )

    max_batches = tc.get("ac_max_batches", None)
    n_epochs = tc.get("ac_epochs", 200)

    for epoch in range(n_epochs):
        metrics = actor_critic_epoch(
            workspace=workspace,
            rssm=rssm,
            actor=actor,
            critic=critic,
            dataloader=seq_loader,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            device=device,
            horizon=ac_cfg.get("imagination_horizon", 15),
            gamma=ac_cfg.get("gamma", 0.997),
            lambda_=ac_cfg.get("lambda_", 0.95),
            entropy_weight=ac_cfg.get("entropy_weight", 3e-4),
            max_batches=max_batches,
        )
        print(
            f"  Epoch {epoch+1:3d}/{n_epochs} — "
            f"actor: {metrics['actor_loss']:.4f}  "
            f"critic: {metrics['critic_loss']:.4f}  "
            f"entropy: {metrics['entropy']:.4f}  "
            f"imag_reward: {metrics['mean_reward']:.4f}"
        )

    return metrics
