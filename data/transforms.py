"""
transforms.py — Preprocessing par domaine
===========================================
Séparé du dataset pour pouvoir swapper facilement.
Ajoute tes augmentations ici.
"""

import torch


def vision_transform(x: torch.Tensor) -> torch.Tensor:
    """Normalise une image [0, 255] uint8 → [0, 1] float32.
    Ajoute tes augmentations (random crop, flip, etc.) ici.
    """
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    return x


def text_transform(tokens: torch.Tensor, max_len: int = 64) -> torch.Tensor:
    """Pad ou tronque une séquence de tokens à max_len."""
    if tokens.size(0) > max_len:
        return tokens[:max_len]
    elif tokens.size(0) < max_len:
        padding = torch.zeros(max_len - tokens.size(0), dtype=tokens.dtype)
        return torch.cat([tokens, padding])
    return tokens


def audio_transform(mel: torch.Tensor, target_len: int = 64) -> torch.Tensor:
    """Pad ou tronque un mel spectrogram à target_len frames."""
    if mel.size(-1) > target_len:
        return mel[..., :target_len]
    elif mel.size(-1) < target_len:
        padding = torch.zeros(*mel.shape[:-1], target_len - mel.size(-1))
        return torch.cat([mel, padding], dim=-1)
    return mel
