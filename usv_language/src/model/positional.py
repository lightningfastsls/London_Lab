"""Sinusoidal positional encoding for transformer sequence modeling.

Standard sin/cos positional encoding from "Attention Is All You Need".
Registered as a buffer so it moves to GPU with the model but isn't a parameter.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding.

    Parameters
    ----------
    d_model:
        Model dimension.
    max_seq_len:
        Maximum sequence length to precompute.
    dropout:
        Dropout applied after adding positional encoding.
    """

    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer: (1, max_seq_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Parameters
        ----------
        x: (B, seq_len, d_model)

        Returns
        -------
        (B, seq_len, d_model) with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
