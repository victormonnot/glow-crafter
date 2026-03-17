"""
dataset.py — Dataset multimodal
=================================
Retourne des tuples alignés (vision, text, audio, ...) pour un même sample.

Deux modes :
  1. DummyMultimodalDataset : données aléatoires pour tester la pipeline
  2. MultimodalDataset      : squelette pour ton vrai dataset

Le point clé : chaque sample contient TOUTES les modalités alignées.
Si une modalité est manquante pour un sample, tu peux :
  - Skip le sample
  - Retourner None et gérer dans le collate_fn
  - Masquer dans la loss
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, Optional

from data.transforms import vision_transform, text_transform, audio_transform


class DummyMultimodalDataset(Dataset):
    """Dataset de test avec données aléatoires.

    Utile pour vérifier que la pipeline tourne de bout en bout
    AVANT de brancher un vrai dataset. Si ça converge pas sur du
    random, c'est que t'as un bug dans l'archi/les losses.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        image_size: int = 32,
        image_channels: int = 3,
        vocab_size: int = 10000,
        seq_len: int = 64,
        n_mels: int = 64,
        audio_len: int = 64,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.image_channels = image_channels
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_mels = n_mels
        self.audio_len = audio_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Seed basé sur l'index pour reproductibilité
        gen = torch.Generator().manual_seed(idx)

        return {
            "vision": torch.rand(
                self.image_channels, self.image_size, self.image_size, generator=gen
            ),
            "text": torch.randint(
                0, self.vocab_size, (self.seq_len,), generator=gen
            ),
            "audio": torch.randn(
                self.n_mels, self.audio_len, generator=gen
            ),
        }


class MultimodalDataset(Dataset):
    """Squelette pour ton vrai dataset multimodal.

    TODO: Implémente __getitem__ pour retourner tes vrais samples alignés.
    Exemples de datasets réels :
      - COCO Captions (image + texte)
      - AudioCaps (audio + texte)
      - HowTo100M (vidéo + audio + texte)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        max_seq_len: int = 64,
        audio_target_len: int = 64,
    ):
        self.data_root = data_root
        self.split = split
        self.max_seq_len = max_seq_len
        self.audio_target_len = audio_target_len

        # TODO: Charge ton index ici
        # self.samples = load_index(data_root, split)
        self.samples = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        result = {}

        # Vision
        if "image_path" in sample:
            # img = load_image(sample["image_path"])
            # result["vision"] = vision_transform(img)
            pass

        # Text
        if "tokens" in sample:
            # tokens = torch.tensor(sample["tokens"])
            # result["text"] = text_transform(tokens, self.max_seq_len)
            pass

        # Audio
        if "mel_path" in sample:
            # mel = load_mel(sample["mel_path"])
            # result["audio"] = audio_transform(mel, self.audio_target_len)
            pass

        return result


def get_dataset(config: dict) -> Dataset:
    """Factory : retourne le bon dataset selon la config."""
    name = config["data"]["dataset"]

    if name == "dummy":
        return DummyMultimodalDataset(
            image_size=config["vision"]["input_size"],
            image_channels=config["vision"]["input_channels"],
            vocab_size=config["text"]["vocab_size"],
            seq_len=config["text"]["max_seq_len"],
            n_mels=config["audio"]["n_mels"],
        )
    else:
        raise ValueError(
            f"Dataset '{name}' pas implémenté. "
            f"Ajoute-le dans dataset.py ou utilise 'dummy' pour tester."
        )
