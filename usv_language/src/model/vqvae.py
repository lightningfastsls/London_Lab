"""Full VQ-VAE model: Encoder → VQ → Transformer → Decoder + next-code prediction.

Assembles all components into a single model with:
- Column-wise encoding of spectrogram frequency vectors
- Vector quantization to discrete codebook entries
- Causal transformer for sequential modeling of code sequences
- Column-wise decoding back to spectrogram
- Next-code prediction head for autoregressive training
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .encoder import ColumnEncoder
from .decoder import ColumnDecoder
from .quantizer import VectorQuantizer
from .transformer import CausalTransformer


@dataclass(frozen=True)
class VQVAEConfig:
    """Full model hyperparameters."""

    # Input
    n_freq: int = 170          # Number of frequency bins

    # Encoder/Decoder
    d_model: int = 64          # Latent dimension
    encoder_hidden_dims: tuple = (256, 128)
    decoder_hidden_dims: tuple = (128, 256)
    enc_dec_dropout: float = 0.1

    # Vector Quantizer
    codebook_size: int = 512
    ema_decay: float = 0.99
    commitment_weight: float = 0.25
    dead_code_threshold: int = 2
    vq_epsilon: float = 1e-5

    # Transformer
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    transformer_dropout: float = 0.1
    max_seq_len: int = 512

    def __post_init__(self) -> None:
        if self.n_freq <= 0:
            raise ValueError("n_freq must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.codebook_size <= 0:
            raise ValueError("codebook_size must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")


class VQVAE(nn.Module):
    """VQ-VAE with Transformer backbone for USV language modeling.

    Forward pipeline:
        spectrogram → encoder → quantizer → transformer → decoder → reconstruction
                                    ↓              ↓
                                indices    next_code_logits

    Parameters
    ----------
    config:
        VQVAEConfig with all hyperparameters.
    """

    def __init__(self, config: VQVAEConfig) -> None:
        super().__init__()
        self.config = config

        self.encoder = ColumnEncoder(
            n_freq=config.n_freq,
            d_model=config.d_model,
            hidden_dims=list(config.encoder_hidden_dims),
            dropout=config.enc_dec_dropout,
        )

        self.quantizer = VectorQuantizer(
            codebook_size=config.codebook_size,
            d_model=config.d_model,
            ema_decay=config.ema_decay,
            commitment_weight=config.commitment_weight,
            dead_code_threshold=config.dead_code_threshold,
            epsilon=config.vq_epsilon,
        )

        self.transformer = CausalTransformer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.transformer_dropout,
            max_seq_len=config.max_seq_len,
        )

        self.decoder = ColumnDecoder(
            d_model=config.d_model,
            n_freq=config.n_freq,
            hidden_dims=list(config.decoder_hidden_dims),
            dropout=config.enc_dec_dropout,
        )

        # Next-code prediction head: predict code index at position t+1
        self.next_code_head = nn.Linear(config.d_model, config.codebook_size)

    def forward(self, x: torch.Tensor) -> dict:
        """Full forward pass.

        Parameters
        ----------
        x: (B, seq_len, n_freq) — normalized spectrogram chunk

        Returns
        -------
        Dict with keys:
            x_recon: (B, seq_len, n_freq) — reconstructed spectrogram
            vq_loss: scalar — VQ commitment loss
            indices: (B, seq_len) — codebook indices
            next_code_logits: (B, seq_len, codebook_size) — next-code predictions
            perplexity: scalar — codebook usage diversity
            codebook_usage: scalar — fraction of codes used
        """
        # Encode
        z = self.encoder(x)  # (B, S, d_model)

        # Quantize
        vq_out = self.quantizer(z)  # z_q, indices, vq_loss, ...
        z_q = vq_out["z_q"]  # (B, S, d_model)

        # Transform (causal)
        h = self.transformer(z_q)  # (B, S, d_model)

        # Decode
        x_recon = self.decoder(h)  # (B, S, n_freq)

        # Next-code prediction
        next_code_logits = self.next_code_head(h)  # (B, S, codebook_size)

        return {
            "x_recon": x_recon,
            "vq_loss": vq_out["vq_loss"],
            "indices": vq_out["indices"],
            "next_code_logits": next_code_logits,
            "perplexity": vq_out["perplexity"],
            "codebook_usage": vq_out["codebook_usage"],
        }

    @torch.no_grad()
    def encode_to_codes(self, x: torch.Tensor) -> torch.LongTensor:
        """Encode spectrogram to discrete code sequence.

        Parameters
        ----------
        x: (B, seq_len, n_freq)

        Returns
        -------
        (B, seq_len) LongTensor of codebook indices.
        """
        z = self.encoder(x)
        vq_out = self.quantizer(z)
        return vq_out["indices"]

    @torch.no_grad()
    def decode_codes_to_spectrogram(self, codes: torch.LongTensor) -> torch.Tensor:
        """Decode discrete codes back to spectrogram.

        Parameters
        ----------
        codes: (B, seq_len) LongTensor of codebook indices

        Returns
        -------
        (B, seq_len, n_freq) reconstructed spectrogram
        """
        z_q = self.quantizer.decode_indices(codes)  # (B, S, d_model)
        h = self.transformer(z_q)
        return self.decoder(h)

    @torch.no_grad()
    def generate(
        self,
        prompt_codes: torch.LongTensor,
        max_new: int = 100,
        temperature: float = 1.0,
    ) -> torch.LongTensor:
        """Autoregressive generation from a prompt sequence.

        Parameters
        ----------
        prompt_codes: (1, prompt_len) — seed code sequence
        max_new: Number of new codes to generate.
        temperature: Sampling temperature.

        Returns
        -------
        (1, prompt_len + max_new) — generated code sequence.
        """
        codes = prompt_codes.clone()

        for _ in range(max_new):
            # Truncate to max_seq_len if needed
            context = codes[:, -self.config.max_seq_len:]
            z_q = self.quantizer.decode_indices(context)
            h = self.transformer(z_q)

            # Predict next code from last position
            logits = self.next_code_head(h[:, -1:, :])  # (1, 1, K)
            logits = logits.squeeze(1) / temperature  # (1, K)

            probs = torch.softmax(logits, dim=-1)
            next_code = torch.multinomial(probs, 1)  # (1, 1)
            codes = torch.cat([codes, next_code], dim=1)

        return codes
