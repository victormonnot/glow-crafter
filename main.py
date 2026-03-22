"""
main.py — Point d'entree GLoW Crafter
=======================================
Chaque phase sauvegarde son propre checkpoint (versione).
On peut relancer depuis n'importe quelle phase.

Usage:
    python main.py                     # Toutes les phases (1→2→3→4)
    python main.py --phase 1           # Phase 1 seulement
    python main.py --phase 2           # Phase 2 (charge checkpoint phase 1)
    python main.py --phase 3           # Phase 3 (charge checkpoint phase 2)
    python main.py --phase 4           # Phase 4 (charge checkpoint phase 3)
    python main.py --phase 3 4         # Phases 3 et 4
    python main.py --eval              # Eval seulement (charge derniers checkpoints)
    python main.py --skip-collect      # Skip la collecte d'episodes
    python main.py --list-checkpoints  # Liste tous les checkpoints avec metriques
    python main.py --load-version phase2_gw_v1  # Charge une version specifique
"""

import argparse
import glob
import os
import re
from datetime import datetime

import torch
import yaml
from torch.utils.data import DataLoader, random_split

from data.collector import CrafterCollector
from data.dataset import CrafterTransitionDataset, CrafterSequenceDataset
from models.actor_critic import Actor, Critic
from models.decoders import StateDecoder, VisionDecoder
from models.domain_module import DomainModule
from models.encoders import StateEncoder, VisionEncoder
from models.rssm import RSSM
from models.workspace import GlobalWorkspace
from pipeline.eval import evaluate, evaluate_crafter_agent
from pipeline.train import train_phase1, train_phase2, train_phase3, train_phase4


CHECKPOINT_DIR = "checkpoints"


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _next_version(name: str) -> int:
    """Find the next version number for a checkpoint name."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    existing = glob.glob(os.path.join(CHECKPOINT_DIR, f"{name}_v*.pt"))
    versions = []
    for f in existing:
        m = re.search(rf"{re.escape(name)}_v(\d+)\.pt$", f)
        if m:
            versions.append(int(m.group(1)))
    return max(versions, default=0) + 1


def save_checkpoint(name: str, metrics: dict = None, **kwargs):
    """Sauvegarde un checkpoint versione avec metriques.

    Cree name_vN.pt et met a jour name_latest.pt (copie).
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    version = _next_version(name)

    kwargs["_meta"] = {
        "name": name,
        "version": version,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics or {},
    }

    versioned_path = os.path.join(CHECKPOINT_DIR, f"{name}_v{version}.pt")
    latest_path = os.path.join(CHECKPOINT_DIR, f"{name}_latest.pt")

    torch.save(kwargs, versioned_path)
    torch.save(kwargs, latest_path)
    print(f"  Checkpoint sauvegarde: {versioned_path}")


def load_checkpoint(name: str, device: torch.device, version: str = None) -> dict:
    """Charge un checkpoint. Par defaut charge la version latest."""
    if version:
        path = os.path.join(CHECKPOINT_DIR, f"{version}.pt")
    else:
        path = os.path.join(CHECKPOINT_DIR, f"{name}_latest.pt")
        # Backward compat: try old format (name.pt) if latest doesn't exist
        if not os.path.exists(path):
            path = os.path.join(CHECKPOINT_DIR, f"{name}.pt")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint introuvable: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    meta = ckpt.get("_meta", {})
    v_str = f"v{meta.get('version', '?')}" if meta else "?"
    print(f"  Checkpoint charge: {path} ({v_str})")
    return ckpt


def list_checkpoints():
    """Affiche tous les checkpoints avec leurs metriques."""
    if not os.path.exists(CHECKPOINT_DIR):
        print("Pas de checkpoints.")
        return

    files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.pt")))
    # Skip _latest duplicates
    files = [f for f in files if "_latest.pt" not in f]
    if not files:
        print("Pas de checkpoints.")
        return

    print(f"\n{'File':<35} {'Date':<22} {'Metrics'}")
    print("-" * 100)

    for f in files:
        basename = os.path.basename(f)
        try:
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
            meta = ckpt.get("_meta", {})
            ts = meta.get("timestamp", "?")
            metrics = meta.get("metrics", {})
            metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                                    if isinstance(v, (int, float)))
            print(f"  {basename:<33} {ts:<22} {metrics_str}")
        except Exception as e:
            print(f"  {basename:<33} (erreur: {e})")

    print()


def build_workspace(config: dict) -> GlobalWorkspace:
    """Construit le Global Workspace avec les 3 domain modules."""
    cfg_proj = config["projection"]
    ws_dim = config["workspace_dim"]
    modules = {}

    cfg_v = config["vision"]
    modules["vision"] = DomainModule(
        name="vision",
        encoder=VisionEncoder(
            input_channels=cfg_v["input_channels"],
            hidden_dim=cfg_v["hidden_dim"],
            latent_dim=config["latent_dims"]["vision"],
        ),
        decoder=VisionDecoder(
            latent_dim=config["latent_dims"]["vision"],
            hidden_dim=cfg_v["hidden_dim"],
            output_channels=cfg_v["input_channels"],
        ),
        latent_dim=config["latent_dims"]["vision"],
        workspace_dim=ws_dim,
        proj_hidden_dim=cfg_proj["hidden_dim"],
        proj_num_layers=cfg_proj["num_layers"],
        proj_dropout=cfg_proj["dropout"],
    )

    cfg_s = config["state"]
    modules["state"] = DomainModule(
        name="state",
        encoder=StateEncoder(
            input_dim=cfg_s["input_dim"],
            hidden_dim=cfg_s["hidden_dim"],
            latent_dim=config["latent_dims"]["state"],
        ),
        decoder=StateDecoder(
            latent_dim=config["latent_dims"]["state"],
            hidden_dim=cfg_s["hidden_dim"],
            output_dim=cfg_s["input_dim"],
        ),
        latent_dim=config["latent_dims"]["state"],
        workspace_dim=ws_dim,
        proj_hidden_dim=cfg_proj["hidden_dim"],
        proj_num_layers=cfg_proj["num_layers"],
        proj_dropout=cfg_proj["dropout"],
    )

    return GlobalWorkspace(
        domain_modules=modules,
        workspace_dim=ws_dim,
        fusion=config["workspace"]["fusion"],
        num_heads=config["workspace"]["num_heads"],
        num_layers=config["workspace"]["num_layers"],
    )


def build_rssm(config: dict) -> RSSM:
    rssm_cfg = config.get("rssm", {})
    return RSSM(
        workspace_dim=config["workspace_dim"],
        num_actions=config["crafter"]["num_actions"],
        action_dim=rssm_cfg.get("action_dim", 32),
        deter_dim=rssm_cfg.get("deter_dim", 256),
        stoch_dim=rssm_cfg.get("stoch_dim", 64),
        hidden_dim=rssm_cfg.get("hidden_dim", 256),
    )


def build_actor_critic(config: dict, rssm: RSSM):
    ac_cfg = config.get("actor_critic", {})
    state_dim = rssm.state_dim
    actor = Actor(
        state_dim=state_dim,
        hidden_dim=ac_cfg.get("hidden_dim", 256),
        num_actions=config["crafter"]["num_actions"],
    )
    critic = Critic(
        state_dim=state_dim,
        hidden_dim=ac_cfg.get("hidden_dim", 256),
    )
    return actor, critic


def _load_gw(workspace, device, version=None):
    """Try loading GW checkpoint (phase2 then phase1)."""
    try:
        ckpt = load_checkpoint("phase2_gw", device, version=version)
        workspace.load_state_dict(ckpt["workspace_state_dict"])
        return True
    except FileNotFoundError:
        try:
            ckpt = load_checkpoint("phase1_gw", device, version=version)
            workspace.load_state_dict(ckpt["workspace_state_dict"])
            return True
        except FileNotFoundError:
            return False


def main():
    parser = argparse.ArgumentParser(description="GLoW Crafter Training")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--phase", type=int, nargs="*", default=None,
                        help="Phase(s) a executer (1, 2, 3, 4). Par defaut: toutes.")
    parser.add_argument("--eval", action="store_true", help="Eval seulement")
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--list-checkpoints", action="store_true",
                        help="Liste tous les checkpoints avec metriques")
    parser.add_argument("--load-version", type=str, default=None,
                        help="Version specifique a charger (ex: phase2_gw_v1)")
    args = parser.parse_args()

    if args.list_checkpoints:
        list_checkpoints()
        return

    config = load_config(args.config)
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Quelles phases executer
    if args.eval:
        phases = []
    elif args.phase is not None:
        phases = args.phase
    else:
        phases = [1, 2, 3, 4]

    # --- Phase 0 : Data collection ---
    tc = config["training"]
    data_dir = tc["data_dir"]

    if not args.skip_collect and not os.path.exists(data_dir) and not args.eval:
        print("=" * 60)
        print("PHASE 0 — Collecte d'episodes Crafter")
        print("=" * 60)
        collector = CrafterCollector(
            save_dir=data_dir,
            num_episodes=tc["collect_episodes"],
            seed=tc["collect_seed"],
        )
        collector.collect()
        print()
    elif os.path.exists(data_dir):
        n_eps = len([f for f in os.listdir(data_dir) if f.endswith(".npz")])
        print(f"Episodes deja collectes: {n_eps} dans {data_dir}")

    # --- Build models ---
    workspace = build_workspace(config).to(device)
    rssm = build_rssm(config).to(device)
    actor, critic = build_actor_critic(config, rssm)
    actor = actor.to(device)
    critic = critic.to(device)

    print(f"GW: {sum(p.numel() for p in workspace.parameters()):,} params")
    print(f"RSSM: {sum(p.numel() for p in rssm.parameters()):,} params")
    print(f"Actor: {sum(p.numel() for p in actor.parameters()):,} params")
    print(f"Critic: {sum(p.numel() for p in critic.parameters()):,} params")
    print()

    # Version specifique a charger ?
    load_ver = args.load_version

    # --- Charger checkpoints des phases precedentes ---
    if 1 not in phases:
        if not _load_gw(workspace, device, version=load_ver):
            if phases:
                print("WARN: Pas de checkpoint GW trouve, on part de zero.")

    if 3 not in phases and any(p >= 4 for p in phases):
        try:
            ckpt = load_checkpoint("phase3_rssm", device, version=load_ver)
            rssm.load_state_dict(ckpt["rssm_state_dict"])
        except FileNotFoundError:
            print("WARN: Pas de checkpoint RSSM trouve.")

    if args.eval:
        _load_gw(workspace, device, version=load_ver)
        try:
            ckpt = load_checkpoint("phase3_rssm", device, version=load_ver)
            rssm.load_state_dict(ckpt["rssm_state_dict"])
        except FileNotFoundError:
            pass
        try:
            ckpt = load_checkpoint("phase4_agent", device, version=load_ver)
            actor.load_state_dict(ckpt["actor_state_dict"])
            critic.load_state_dict(ckpt["critic_state_dict"])
        except FileNotFoundError:
            pass

    # --- Data loaders ---
    needs_transitions = any(p in phases for p in [1, 2]) or args.eval
    needs_sequences = any(p in phases for p in [3, 4])

    train_loader = eval_loader = seq_loader = None

    if needs_transitions and os.path.exists(data_dir):
        trans_ds = CrafterTransitionDataset(data_dir)
        train_sz = int(len(trans_ds) * config["data"]["train_split"])
        train_ds, eval_ds = random_split(trans_ds, [train_sz, len(trans_ds) - train_sz])
        train_loader = DataLoader(
            train_ds, batch_size=tc["batch_size"], shuffle=True,
            num_workers=config["data"]["num_workers"], pin_memory=True,
        )
        eval_loader = DataLoader(
            eval_ds, batch_size=tc["batch_size"], shuffle=False,
            num_workers=config["data"]["num_workers"],
        )

    if needs_sequences and os.path.exists(data_dir):
        seq_ds = CrafterSequenceDataset(data_dir, seq_len=tc.get("wm_seq_len", 50))
        seq_loader = DataLoader(
            seq_ds, batch_size=tc.get("wm_batch_size", 32), shuffle=True,
            num_workers=config["data"]["num_workers"], pin_memory=True,
        )

    # --- Execute phases ---
    if 1 in phases:
        print()
        metrics = train_phase1(workspace, train_loader, config, device)
        save_checkpoint("phase1_gw",
                        metrics=metrics,
                        workspace_state_dict=workspace.state_dict(),
                        config=config)

    if 2 in phases:
        if 1 not in phases:
            try:
                ckpt = load_checkpoint("phase1_gw", device)
                workspace.load_state_dict(ckpt["workspace_state_dict"])
            except FileNotFoundError:
                print("WARN: Pas de checkpoint phase 1, on continue avec les poids actuels.")

        print()
        metrics = train_phase2(workspace, train_loader, config, device)

        # Add eval metrics
        eval_metrics = {}
        if eval_loader is not None:
            print()
            eval_metrics = evaluate(workspace, eval_loader, config, device)

        all_metrics = {**metrics, **eval_metrics}
        save_checkpoint("phase2_gw",
                        metrics=all_metrics,
                        workspace_state_dict=workspace.state_dict(),
                        config=config)

    if 3 in phases:
        if 2 not in phases:
            try:
                ckpt = load_checkpoint("phase2_gw", device)
                workspace.load_state_dict(ckpt["workspace_state_dict"])
            except FileNotFoundError:
                print("WARN: Pas de checkpoint phase 2.")

        print()
        metrics = train_phase3(workspace, rssm, seq_loader, config, device)
        save_checkpoint("phase3_rssm",
                        metrics=metrics,
                        rssm_state_dict=rssm.state_dict(),
                        config=config)

    if 4 in phases:
        if 3 not in phases:
            _load_gw(workspace, device)
            try:
                ckpt = load_checkpoint("phase3_rssm", device)
                rssm.load_state_dict(ckpt["rssm_state_dict"])
            except FileNotFoundError:
                print("WARN: Pas de checkpoint phase 3.")

        print()
        metrics = train_phase4(workspace, rssm, actor, critic, seq_loader, config, device)

        # Eval agent
        eval_cfg = config.get("eval", {})
        agent_results = evaluate_crafter_agent(
            workspace, rssm, actor, device,
            num_episodes=eval_cfg.get("eval_episodes", 20),
        )
        all_metrics = {**metrics, "mean_reward": agent_results["mean_reward"],
                       "achievements": agent_results["unique_achievements"]}
        save_checkpoint("phase4_agent",
                        metrics=all_metrics,
                        actor_state_dict=actor.state_dict(),
                        critic_state_dict=critic.state_dict(),
                        config=config)

    if args.eval:
        if eval_loader is not None:
            evaluate(workspace, eval_loader, config, device)
        eval_cfg = config.get("eval", {})
        evaluate_crafter_agent(
            workspace, rssm, actor, device,
            num_episodes=eval_cfg.get("eval_episodes", 20),
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
