"""Column-wise encoder: compresses each spectrogram column independently.

Architecture: MLP with LayerNorm and GELU activations.
Input:  (B, seq_len, n_freq)  — one frequency vector per time column
Output: (B, seq_len, d_model) — compressed latent representation
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ColumnEncoder(nn.Module):
    """MLP encoder that maps each frequency column to a latent vector.

    Parameters
    ----------
    n_freq:
        Number of frequency bins per column (input dimension).
    d_model:
        Latent dimension (output dimension).
    hidden_dims:
        List of hidden layer dimensions. Default: [256, 128].
    dropout:
        Dropout rate between layers.
    """

    def __init__(
        self,
        n_freq: int,
        d_model: int,
        hidden_dims: List[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers: list[nn.Module] = []
        in_dim = n_freq
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        # Final projection: no activation, just linear + norm
        layers.extend([
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode frequency columns to latent vectors.

        Parameters
        ----------
        x: (B, seq_len, n_freq)

        Returns
        -------
        (B, seq_len, d_model)
        """
        return self.net(x)
