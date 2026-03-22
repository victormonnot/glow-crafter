"""
transforms.py — Preprocessing par domaine (Crafter)
=====================================================
"""

import torch


def vision_transform(x: torch.Tensor) -> torch.Tensor:
    """Normalise une image [0, 255] uint8 → [0, 1] float32."""
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    return x


def state_transform(state: torch.Tensor) -> torch.Tensor:
    """State vector deja normalise par le collector, pass-through."""
    return state.float()


def action_transform(action: int) -> torch.Tensor:
    """Convertit un int action en tensor."""
    return torch.tensor(action, dtype=torch.long)
