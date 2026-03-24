"""Autoregressive spectrogram transformer for USV next-column prediction.

Architecture: Pre-norm transformer with learned positional embeddings.
Input: (batch, seq_len, n_freq) spectrogram columns.
Output: (batch, seq_len, n_freq) predicted next columns.

The model uses causal masking so each position can only attend to itself
and earlier positions — the standard autoregressive constraint.

Design choices:
- Pre-norm (LayerNorm before attention/FFN) for stable training gradients.
- GELU activation in FFN (smoother gradient landscape than ReLU).
- Learned positional embeddings (simpler than sinusoidal, works well for
  fixed max_seq_len).
- ~25M parameters: 8 layers × (512 d_model, 2048 d_ffn, 8 heads).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TransformerConfig:
    """Configuration for the spectrogram autoregressive transformer.

    Attributes
    ----------
    n_freq:
        Number of frequency bins (20-120 kHz at sr=300k, n_fft=512 → 170).
    d_model:
        Model hidden dimension.
    n_heads:
        Number of attention heads (d_head = d_model // n_heads = 64).
    n_layers:
        Number of transformer blocks.
    d_ffn:
        FFN hidden dimension (typically 4 × d_model).
    max_seq_len:
        Maximum sequence length (causal mask size).
    dropout:
        Dropout rate for attention and FFN.
    """

    n_freq: int = 170
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ffn: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )
        if self.n_freq <= 0:
            raise ValueError(f"n_freq must be positive, got {self.n_freq}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.d_ffn <= 0:
            raise ValueError(f"d_ffn must be positive, got {self.d_ffn}")
        if self.max_seq_len <= 0:
            raise ValueError(
                f"max_seq_len must be positive, got {self.max_seq_len}"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(
                f"dropout must be in [0, 1), got {self.dropout}"
            )


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with causal self-attention.

    Architecture per block:
        x -> LayerNorm -> MultiheadAttention -> + residual
          -> LayerNorm -> FFN(Linear-GELU-Linear) -> + residual
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through one transformer block.

        Parameters
        ----------
        x:
            Input tensor, shape (batch, seq_len, d_model).
        causal_mask:
            Boolean mask (seq_len, seq_len). True = do not attend.
        padding_mask:
            Boolean mask (batch, seq_len). True = padding position.
        """
        # Pre-norm self-attention + residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
        )
        x = x + self.dropout1(attn_out)

        # Pre-norm FFN + residual
        x = x + self.ffn(self.norm2(x))
        return x


class SpectrogramTransformer(nn.Module):
    """Autoregressive transformer for next-column spectrogram prediction.

    Takes a sequence of spectrogram columns and predicts the next column
    at each position. Uses causal masking to enforce the autoregressive
    constraint (each position only sees itself and past).

    Parameters
    ----------
    config:
        Model configuration (dimensions, layers, etc.).
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        # Input projection: n_freq -> d_model with GELU + LayerNorm
        self.input_proj = nn.Sequential(
            nn.Linear(config.n_freq, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )

        # Learned positional embeddings
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

        # Input dropout
        self.input_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model, config.n_heads, config.d_ffn, config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        # Output projection: d_model -> n_freq
        self.output_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.n_freq),
        )

        # Causal mask: upper-triangular True = do not attend to future
        causal = torch.triu(
            torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1,
        ).bool()
        self.register_buffer("causal_mask", causal)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Forward pass.

        Parameters
        ----------
        x:
            Input spectrogram columns, shape (batch, seq_len, n_freq).
        attention_mask:
            Padding mask, shape (batch, seq_len). True where padding.
            Passed as key_padding_mask to attention layers.
        return_hidden_states:
            If True, return hidden states from all layers.

        Returns
        -------
        output:
            Predicted next columns, shape (batch, seq_len, n_freq).
        hidden_states:
            List of (batch, seq_len, d_model) tensors, one per layer.
            None if return_hidden_states=False.
        """
        B, S, _ = x.shape

        # Project input to model dimension
        h = self.input_proj(x)  # (B, S, d_model)

        # Add learned positional embeddings
        positions = torch.arange(S, device=x.device)
        h = h + self.pos_embed(positions)  # broadcasts over batch

        h = self.input_dropout(h)

        # Slice causal mask to actual sequence length
        causal = self.causal_mask[:S, :S]

        # Run through transformer blocks
        hidden_states = [] if return_hidden_states else None
        for block in self.blocks:
            h = block(h, causal, attention_mask)
            if return_hidden_states:
                hidden_states.append(h)

        # Project back to frequency space
        output = self.output_head(h)  # (B, S, n_freq)

        return output, hidden_states

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
