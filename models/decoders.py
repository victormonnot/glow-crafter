"""
decoders.py — Décodeurs domaine-spécifiques
=============================================
Miroir des encodeurs. Prend un vecteur latent z_d et reconstruit
l'entrée brute du domaine.

Utilisés pour :
  1. La loss de reconstruction (autoencoder) en phase 1
  2. La traduction cross-modale : workspace → domaine cible
"""

import torch
import torch.nn as nn


class VisionDecoder(nn.Module):
    """z_vision → Image reconstruite."""

    def __init__(self, latent_dim: int = 256, hidden_dim: int = 128, output_channels: int = 3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim * 4 * 4 * 4)
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            # 4x4 → 8x8
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            # 8x8 → 16x16
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            # 16x16 → 32x32
            nn.ConvTranspose2d(hidden_dim, output_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),  # Images normalisées [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → x_hat: (B, C, 32, 32)"""
        h = self.fc(z).view(-1, self.hidden_dim * 4, 4, 4)
        return self.net(h)


class TextDecoder(nn.Module):
    """z_text → Logits sur le vocabulaire pour chaque position.

    Simplifié : on génère toute la séquence en parallèle (non-autorégressif).
    Pour un vrai projet, passe en autorégressif avec masque causal.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        embed_dim: int = 128,
        vocab_size: int = 10000,
        max_seq_len: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.latent_to_seq = nn.Linear(latent_dim, max_seq_len * embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → logits: (B, seq_len, vocab_size)"""
        B = z.size(0)
        h = self.latent_to_seq(z).view(B, self.max_seq_len, -1)
        positions = torch.arange(self.max_seq_len, device=z.device).unsqueeze(0).expand(B, -1)
        h = h + self.pos_embedding(positions)
        h = self.transformer(h)
        return self.head(h)


class AudioDecoder(nn.Module):
    """z_audio → Mel spectrogram reconstruit."""

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 128, n_mels: int = 64, output_len: int = 64):
        super().__init__()
        self.output_len = output_len
        self.fc = nn.Linear(latent_dim, hidden_dim * 2 * (output_len // 4))
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, n_mels, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → x_hat: (B, n_mels, T)"""
        h = self.fc(z).view(-1, self.hidden_dim * 2, self.output_len // 4)
        return self.net(h)
