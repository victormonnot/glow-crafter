"""
main.py — Point d'entrée GLoW Minimal
=======================================
Assemble tous les composants et lance l'entraînement.

Usage:
    python main.py                          # Config par défaut
    python main.py --config path/to/cfg.yaml  # Config custom

C'est ici que tu branches tes vrais composants :
  - Swap DummyDataset → ton dataset
  - Swap les encodeurs CNN → ResNet/ViT
  - Ajuste les hyperparams dans config.yaml
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader, random_split

from models.encoders import VisionEncoder, TextEncoder, AudioEncoder
from models.decoders import VisionDecoder, TextDecoder, AudioDecoder
from models.domain_module import DomainModule
from models.workspace import GlobalWorkspace
from data.dataset import get_dataset
from pipeline.train import train
from pipeline.eval import evaluate


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_domain_modules(config: dict) -> dict:
    """Construit les domain modules selon la config.

    C'est ici que tu swappes les architectures.
    Chaque domaine = (encoder, decoder, latent_dim).
    """
    cfg_proj = config["projection"]
    ws_dim = config["workspace_dim"]

    modules = {}

    # --- Vision ---
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

    # --- Text ---
    cfg_t = config["text"]
    modules["text"] = DomainModule(
        name="text",
        encoder=TextEncoder(
            vocab_size=cfg_t["vocab_size"],
            embed_dim=cfg_t["embed_dim"],
            latent_dim=config["latent_dims"]["text"],
            max_seq_len=cfg_t["max_seq_len"],
        ),
        decoder=TextDecoder(
            latent_dim=config["latent_dims"]["text"],
            embed_dim=cfg_t["embed_dim"],
            vocab_size=cfg_t["vocab_size"],
            max_seq_len=cfg_t["max_seq_len"],
        ),
        latent_dim=config["latent_dims"]["text"],
        workspace_dim=ws_dim,
        proj_hidden_dim=cfg_proj["hidden_dim"],
        proj_num_layers=cfg_proj["num_layers"],
        proj_dropout=cfg_proj["dropout"],
    )

    # --- Audio ---
    cfg_a = config["audio"]
    modules["audio"] = DomainModule(
        name="audio",
        encoder=AudioEncoder(
            n_mels=cfg_a["n_mels"],
            hidden_dim=cfg_a["hidden_dim"],
            latent_dim=config["latent_dims"]["audio"],
        ),
        decoder=AudioDecoder(
            latent_dim=config["latent_dims"]["audio"],
            hidden_dim=cfg_a["hidden_dim"],
            n_mels=cfg_a["n_mels"],
        ),
        latent_dim=config["latent_dims"]["audio"],
        workspace_dim=ws_dim,
        proj_hidden_dim=cfg_proj["hidden_dim"],
        proj_num_layers=cfg_proj["num_layers"],
        proj_dropout=cfg_proj["dropout"],
    )

    return modules


def main():
    parser = argparse.ArgumentParser(description="GLoW Minimal Training")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config"
    )
    args = parser.parse_args()

    # --- Config ---
    config = load_config(args.config)
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Dataset ---
    dataset = get_dataset(config)
    train_size = int(len(dataset) * config["data"]["train_split"])
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
    )

    # --- Modèle ---
    domain_modules = build_domain_modules(config)
    workspace = GlobalWorkspace(
        domain_modules=domain_modules,
        workspace_dim=config["workspace_dim"],
        fusion=config["workspace"]["fusion"],
        num_heads=config["workspace"]["num_heads"],
        num_layers=config["workspace"]["num_layers"],
    ).to(device)

    # Compte les params
    total_params = sum(p.numel() for p in workspace.parameters())
    trainable_params = sum(p.numel() for p in workspace.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable:    {trainable_params:,}")
    print()

    # --- Train ---
    train(workspace, train_loader, config, device)

    # --- Eval ---
    results = evaluate(workspace, eval_loader, config, device)

    # --- Save ---
    save_path = "checkpoint.pt"
    torch.save({
        "model_state_dict": workspace.state_dict(),
        "config": config,
        "eval_results": results,
    }, save_path)
    print(f"\nCheckpoint sauvegardé: {save_path}")


if __name__ == "__main__":
    main()
