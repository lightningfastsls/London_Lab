"""Transformer suffix computation: decode hidden states through remaining layers.

The two-phase architecture (ADR-007) separates the SpectrogramTransformer
(frozen, Phase 8.2) from the HiddenStateVQVAE (frozen, Phase 8.3).  To
convert a codebook entry back to a spectrogram, we need to:

1. Look up the codebook vector (codebook_dim)
2. Decode through VQ-VAE decoder (codebook_dim -> d_model)
3. Run through the *remaining* transformer layers (d_model -> d_model)
4. Apply the output head (d_model -> n_freq)

This module provides that bridge.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def decode_hidden_to_spectrogram(
    transformer: nn.Module,
    hidden_state: torch.Tensor,
    start_layer: int,
) -> torch.Tensor:
    """Run hidden states through remaining transformer layers to get spectrogram.

    Takes hidden states captured at ``start_layer`` (1-indexed) and continues
    the forward pass through the subsequent layers and the output head.

    Parameters
    ----------
    transformer:
        SpectrogramTransformer instance.  Accesses ``blocks``, ``causal_mask``,
        and ``output_head`` attributes.
    hidden_state:
        (B, S, d_model) hidden states from layer ``start_layer``.
    start_layer:
        1-indexed layer where hidden states were captured.  E.g. ``4`` means
        the hidden state is the output of the 4th transformer block.
        Remaining blocks ``[start_layer:]`` (0-indexed) will be applied.

    Returns
    -------
    (B, S, n_freq) predicted spectrogram columns.
    """
    B, S, _ = hidden_state.shape
    h = hidden_state

    # Slice causal mask to actual sequence length
    causal = transformer.causal_mask[:S, :S]

    # Continue through remaining blocks (start_layer is 1-indexed)
    for block in transformer.blocks[start_layer:]:
        h = block(h, causal)

    # Apply output head (LayerNorm + Linear -> n_freq)
    return transformer.output_head(h)


def inject_and_continue(
    transformer: nn.Module,
    vqvae: nn.Module,
    input_spectrogram: torch.Tensor,
    injection_position: int,
    codebook_index: int,
    source_layer: int,
) -> torch.Tensor:
    """Inject a codebook entry at a position and continue the forward pass.

    Runs the transformer forward to ``source_layer``, replaces the hidden
    state at ``injection_position`` with a decoded codebook entry, then
    continues through the remaining layers to produce a spectrogram.

    Parameters
    ----------
    transformer:
        SpectrogramTransformer instance.
    vqvae:
        HiddenStateVQVAE instance (provides quantizer + decoder).
    input_spectrogram:
        (1, S, n_freq) input spectrogram.
    injection_position:
        Time position to inject at (0-indexed).
    codebook_index:
        Which codebook entry to inject (0 to K-1).
    source_layer:
        1-indexed layer at which to inject.

    Returns
    -------
    (1, S, n_freq) output spectrogram with injected concept.
    """
    device = input_spectrogram.device
    B, S, _ = input_spectrogram.shape

    # --- Forward pass up to source_layer ---
    h = transformer.input_proj(input_spectrogram)  # (B, S, d_model)
    positions = torch.arange(S, device=device)
    h = h + transformer.pos_embed(positions)

    causal = transformer.causal_mask[:S, :S]
    for block in transformer.blocks[:source_layer]:
        h = block(h, causal)

    # --- Decode codebook entry to d_model space ---
    code_idx = torch.tensor([codebook_index], device=device, dtype=torch.long)
    code_vec = vqvae.quantizer.decode_indices(code_idx)  # (1, codebook_dim)
    decoded = vqvae.decoder(code_vec)  # (1, d_model)

    # --- Inject at position ---
    h = h.clone()  # avoid in-place modification
    h[:, injection_position, :] = decoded.squeeze(0)

    # --- Continue through remaining layers ---
    for block in transformer.blocks[source_layer:]:
        h = block(h, causal)

    return transformer.output_head(h)
