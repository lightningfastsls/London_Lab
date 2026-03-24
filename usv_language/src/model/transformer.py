"""Causal (GPT-style) Transformer for sequential code prediction.

Pre-norm architecture with causal masking. Processes quantized code sequences
and predicts the next code at each position.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .positional import SinusoidalPositionalEncoding


class CausalTransformerBlock(nn.Module):
    """Single transformer block with pre-norm, causal self-attention, and FFN.

    Parameters
    ----------
    d_model:
        Model dimension.
    n_heads:
        Number of attention heads.
    d_ff:
        Feed-forward hidden dimension.
    dropout:
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal attention.

        Parameters
        ----------
        x: (B, seq_len, d_model)
        causal_mask: (seq_len, seq_len) boolean mask

        Returns
        -------
        (B, seq_len, d_model)
        """
        # Pre-norm self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=causal_mask)
        x = x + attn_out

        # Pre-norm FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class CausalTransformer(nn.Module):
    """Stack of causal transformer blocks with positional encoding.

    Parameters
    ----------
    d_model:
        Model dimension.
    n_heads:
        Number of attention heads.
    n_layers:
        Number of transformer blocks.
    d_ff:
        Feed-forward hidden dimension.
    dropout:
        Dropout rate.
    max_seq_len:
        Maximum sequence length for positional encoding.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList([
            CausalTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process sequence with causal masking.

        Parameters
        ----------
        x: (B, seq_len, d_model)

        Returns
        -------
        (B, seq_len, d_model)
        """
        seq_len = x.size(1)
        # Create causal mask: True where attention is NOT allowed
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, causal_mask)
        return self.final_norm(x)
