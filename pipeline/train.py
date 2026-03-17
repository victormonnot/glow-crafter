"""
train.py — Pipeline d'entraînement GLoW
=========================================
Entraînement en 2 phases :

  Phase 1 (Pretrain) : Chaque domaine s'entraîne comme autoencoder
    indépendant. Le but est d'avoir des encodeurs/décodeurs qui
    marchent AVANT d'aligner les domaines.
    Loss = reconstruction par domaine.

  Phase 2 (Align) : On freeze optionnellement les encodeurs/décodeurs
    et on entraîne les projections + le workspace.
    Loss = contrastive + translation + cycle consistency.

Pourquoi 2 phases ? Parce que si tu alignes des représentations
qui ne veulent rien dire (encodeurs pas entraînés), tu alignes du bruit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from itertools import combinations
from tqdm import tqdm
from typing import Dict, List, Tuple

from models.workspace import GlobalWorkspace
from losses.contrastive import ContrastiveLoss
from losses.translation import TranslationLoss, TextTranslationLoss
from losses.cycle import CycleConsistencyLoss, compute_cycle


def pretrain_epoch(
    workspace: GlobalWorkspace,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Phase 1 : Entraîne chaque domaine comme autoencoder.

    Loss = Σ_d reconstruction_loss(x_d, decode_d(encode_d(x_d)))
    """
    workspace.train()
    total_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device)

        for name, module in workspace.domain_modules.items():
            if name not in batch:
                continue

            x = batch[name].to(device)

            if name == "text":
                # Text : encode → decode → cross-entropy sur les logits
                z = module.encode(x)
                logits = module.decode(z)  # (B, seq_len, vocab_size)
                B, S, V = logits.shape
                loss += F.cross_entropy(logits.reshape(B * S, V), x.reshape(B * S))
            else:
                # Vision, Audio : reconstruction MSE
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
    text_translation_loss_fn: TextTranslationLoss,
    cycle_loss_fn: CycleConsistencyLoss,
) -> Dict[str, float]:
    """Phase 2 : Aligne les domaines dans le workspace.

    Pour chaque paire (A, B) :
      1. Contrastive   : w_a proche de w_b (même sample)
      2. Translation   : x_a → workspace → x̂_b ≈ x_b (et inversement)
      3. Cycle         : z_a → w → z_b → w → ẑ_a ≈ z_a
    """
    workspace.train()
    metrics = {"contrastive": 0.0, "translation": 0.0, "cycle": 0.0, "total": 0.0}

    for batch in dataloader:
        optimizer.zero_grad()

        # Encode tout et projette dans le workspace
        inputs = {name: batch[name].to(device) for name in batch if name in workspace.domain_modules}
        workspace_reprs = workspace.encode_domains(inputs)

        # Aussi besoin des latents (avant projection) pour le cycle
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

            # --- 1. Contrastive ---
            loss_contrastive += contrastive_loss_fn(w_a, w_b)

            # --- 2. Translation (A → B et B → A) ---
            # A → workspace → B
            z_b_hat = mod_b.from_workspace(w_a)
            x_b_hat = mod_b.decode(z_b_hat)

            if name_b == "text":
                loss_translation += text_translation_loss_fn(inputs[name_b], x_b_hat)
            else:
                loss_translation += translation_loss_fn(inputs[name_b], x_b_hat)

            # B → workspace → A
            z_a_hat = mod_a.from_workspace(w_b)
            x_a_hat = mod_a.decode(z_a_hat)

            if name_a == "text":
                loss_translation += text_translation_loss_fn(inputs[name_a], x_a_hat)
            else:
                loss_translation += translation_loss_fn(inputs[name_a], x_a_hat)

            # --- 3. Cycle consistency ---
            z_a_cycle = compute_cycle(mod_a, mod_b, latents[name_a])
            loss_cycle += cycle_loss_fn(latents[name_a].detach(), z_a_cycle)

            z_b_cycle = compute_cycle(mod_b, mod_a, latents[name_b])
            loss_cycle += cycle_loss_fn(latents[name_b].detach(), z_b_cycle)

        # Pondération
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


def train(
    workspace: GlobalWorkspace,
    train_loader: DataLoader,
    config: dict,
    device: torch.device,
):
    """Boucle d'entraînement complète : Phase 1 + Phase 2."""

    tc = config["training"]
    lc = config["losses"]

    # ==================== PHASE 1 : Pretrain ====================
    print("=" * 60)
    print("PHASE 1 — Pretrain autoencoders")
    print("=" * 60)

    optimizer = torch.optim.AdamW(
        workspace.parameters(), lr=tc["pretrain_lr"], weight_decay=tc["weight_decay"]
    )

    for epoch in range(tc["pretrain_epochs"]):
        loss = pretrain_epoch(workspace, train_loader, optimizer, device)
        print(f"  Epoch {epoch+1:3d}/{tc['pretrain_epochs']} — recon loss: {loss:.4f}")

    # ==================== PHASE 2 : Alignment ====================
    print()
    print("=" * 60)
    print("PHASE 2 — Workspace alignment")
    print("=" * 60)

    # Loss functions
    contrastive_fn = ContrastiveLoss(temperature=lc["contrastive"]["temperature"])
    translation_fn = TranslationLoss()
    text_translation_fn = TextTranslationLoss()
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
            text_translation_loss_fn=text_translation_fn,
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

    print()
    print("Training terminé.")
