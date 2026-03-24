"""Column-wise decoder: reconstructs spectrogram columns from latent vectors.

Architecture: MLP with LayerNorm and GELU activations, sigmoid output.
Input:  (B, seq_len, d_model) — latent representation
Output: (B, seq_len, n_freq)  — reconstructed frequency columns in [0, 1]
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ColumnDecoder(nn.Module):
    """MLP decoder that maps latent vectors back to frequency columns.

    Parameters
    ----------
    d_model:
        Latent dimension (input dimension).
    n_freq:
        Number of frequency bins per column (output dimension).
    hidden_dims:
        List of hidden layer dimensions. Default: [128, 256].
    dropout:
        Dropout rate between layers.
    """

    def __init__(
        self,
        d_model: int,
        n_freq: int,
        hidden_dims: List[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 256]

        layers: list[nn.Module] = []
        in_dim = d_model
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        # Final projection with sigmoid for [0, 1] output
        layers.extend([
            nn.Linear(in_dim, n_freq),
            nn.Sigmoid(),
        ])

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to frequency columns.

        Parameters
        ----------
        x: (B, seq_len, d_model)

        Returns
        -------
        (B, seq_len, n_freq) with values in [0, 1]
        """
        return self.net(x)
