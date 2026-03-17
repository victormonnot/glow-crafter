"""
encoders.py — Encodeurs domaine-spécifiques
============================================
Chaque encodeur prend une entrée brute du domaine (image, texte, audio)
et la compresse en un vecteur latent z_d de dimension latent_dim.

C'est du classique. Rien de spécifique à GLoW ici.
Remplace par des architectures plus costauds (ResNet, Transformer, etc.)
quand tu passes à un vrai dataset.
"""

import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """Image → z_vision. CNN basique, swap avec ResNet/ViT quand tu veux."""

    def __init__(self, input_channels: int = 3, hidden_dim: int = 128, latent_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            # 32x32 → 16x16
            nn.Conv2d(input_channels, hidden_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            # 16x16 → 8x8
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            # 8x8 → 4x4
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim * 4, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → z: (B, latent_dim)"""
        return self.net(x)


class TextEncoder(nn.Module):
    """Token IDs → z_text. Embedding + Transformer simplifié."""

    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 128,
        latent_dim: int = 256,
        max_seq_len: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(embed_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len) token ids → z: (B, latent_dim)"""
        B, S = x.shape
        positions = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.embedding(x) + self.pos_embedding(positions)
        h = self.transformer(h)
        # Mean pooling sur la séquence
        h = h.mean(dim=1)
        return self.proj(h)


class AudioEncoder(nn.Module):
    """Mel spectrogram → z_audio. 1D conv sur l'axe temporel."""

    def __init__(self, n_mels: int = 64, hidden_dim: int = 128, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            # (B, n_mels, T) → traité comme (B, channels, length)
            nn.Conv1d(n_mels, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim * 2, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_mels, T) → z: (B, latent_dim)"""
        return self.net(x)
