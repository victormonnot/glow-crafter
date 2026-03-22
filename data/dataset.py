"""
dataset.py — Datasets Crafter pour GLoW
=========================================
Lazy loading : on garde seulement les indices (episode_file, timestep) en RAM.
Les frames sont chargees a la demande depuis les .npz.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict


class CrafterTransitionDataset(Dataset):
    """Transitions individuelles {vision, state, action}, lazy-loaded.

    Ne garde en RAM que la liste (fichier, index_dans_fichier).
    Charge le .npz a chaque __getitem__.
    """

    def __init__(self, data_dir: str):
        self.index = []  # list of (episode_path, timestep)
        episode_files = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))

        if not episode_files:
            raise FileNotFoundError(
                f"Pas d'episodes dans {data_dir}. Lance d'abord la collecte."
            )

        for ep_file in episode_files:
            # Juste lire le nombre de transitions sans charger les arrays
            with np.load(ep_file) as data:
                T = len(data["actions"])
            for t in range(T):
                self.index.append((ep_file, t))

        print(f"CrafterTransitionDataset: {len(self.index)} transitions from {len(episode_files)} episodes")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_file, t = self.index[idx]
        data = np.load(ep_file)

        obs = torch.from_numpy(data["observations"][t].copy()).permute(2, 0, 1).float() / 255.0
        state = torch.from_numpy(data["states"][t].copy())
        action = torch.tensor(int(data["actions"][t]), dtype=torch.long)

        return {
            "vision": obs,      # (3, 64, 64)
            "state": state,     # (16,)
            "action": action,   # scalar
        }


class CrafterSequenceDataset(Dataset):
    """Sequences de longueur fixe, lazy-loaded.

    Garde en RAM : (episode_path, start_index).
    """

    def __init__(self, data_dir: str, seq_len: int = 50):
        self.seq_len = seq_len
        self.index = []  # list of (episode_path, start)
        episode_files = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))

        if not episode_files:
            raise FileNotFoundError(
                f"Pas d'episodes dans {data_dir}. Lance d'abord la collecte."
            )

        for ep_file in episode_files:
            with np.load(ep_file) as data:
                T = len(data["actions"])
            for start in range(max(1, T - seq_len + 1)):
                if start + seq_len <= T:
                    self.index.append((ep_file, start))

        print(f"CrafterSequenceDataset: {len(self.index)} sequences of length {seq_len}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_file, start = self.index[idx]
        data = np.load(ep_file)
        end = start + self.seq_len

        obs = torch.from_numpy(data["observations"][start:end].copy()).permute(0, 3, 1, 2).float() / 255.0
        states = torch.from_numpy(data["states"][start:end].copy())
        actions = torch.from_numpy(data["actions"][start:end].copy()).long()
        rewards = torch.from_numpy(data["rewards"][start:end].copy())
        dones = torch.from_numpy(data["dones"][start:end].astype(np.float32).copy())

        return {
            "vision": obs,       # (L, 3, 64, 64)
            "state": states,     # (L, 16)
            "action": actions,   # (L,)
            "reward": rewards,   # (L,)
            "done": dones,       # (L,)
        }


def get_dataset(config: dict) -> Dataset:
    """Factory : retourne le bon dataset selon la config."""
    name = config["data"]["dataset"]
    data_dir = config["training"]["data_dir"]

    if name == "crafter_transitions":
        return CrafterTransitionDataset(data_dir=data_dir)
    elif name == "crafter_sequences":
        seq_len = config["training"].get("wm_seq_len", 50)
        return CrafterSequenceDataset(data_dir=data_dir, seq_len=seq_len)
    else:
        raise ValueError(
            f"Dataset '{name}' pas implemente. "
            f"Utilise 'crafter_transitions' ou 'crafter_sequences'."
        )
