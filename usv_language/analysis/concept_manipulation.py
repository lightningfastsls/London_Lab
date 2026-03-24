"""Concept injection and manipulation experiments.

Probes what the transformer "thinks" by injecting specific codebook entries
at hidden-state positions and observing the effect on output spectrograms.

Key experiments:
- concept_injection: inject a code at position t, observe future spectrogram
- concept_scan: try every code at a fixed position, compare outputs
- top_k_competing_concepts: find which codes compete at each position
"""

from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from usv_language.analysis.transformer_suffix import (
    decode_hidden_to_spectrogram,
    inject_and_continue,
)


def concept_injection(
    transformer: torch.nn.Module,
    vqvae: torch.nn.Module,
    input_spectrogram: torch.Tensor,
    injection_position: int,
    codebook_index: int,
    source_layer: int,
    n_future_steps: int = 50,
) -> dict:
    """Inject a codebook entry and generate future spectrogram autoregressively.

    1. Inject codebook entry at ``injection_position`` in layer ``source_layer``
    2. Get the full output spectrogram
    3. Generate ``n_future_steps`` additional columns autoregressively

    Parameters
    ----------
    transformer:
        SpectrogramTransformer instance.
    vqvae:
        HiddenStateVQVAE instance.
    input_spectrogram:
        (1, S, n_freq) input spectrogram.
    injection_position:
        Time position to inject at.
    codebook_index:
        Codebook entry to inject.
    source_layer:
        1-indexed injection layer.
    n_future_steps:
        Number of autoregressive continuation steps.

    Returns
    -------
    Dict with:
        injected_output: (S, n_freq) spectrogram with injection
        future_columns: (n_future_steps, n_freq) generated continuation
        original_output: (S, n_freq) spectrogram without injection
    """
    transformer.eval()
    vqvae.eval()
    device = input_spectrogram.device
    n_freq = transformer.config.n_freq
    max_seq_len = transformer.config.max_seq_len

    with torch.no_grad():
        # Original output (no injection)
        original_out, _ = transformer(input_spectrogram)
        original_output = original_out.squeeze(0).cpu().numpy()

        # Injected output
        injected_out = inject_and_continue(
            transformer, vqvae, input_spectrogram,
            injection_position, codebook_index, source_layer,
        )
        injected_output = injected_out.squeeze(0).cpu().numpy()

        # Autoregressive continuation from injected output
        current = injected_out  # (1, S, n_freq)
        future_cols = []
        for _ in range(n_future_steps):
            S = current.shape[1]
            if S >= max_seq_len:
                # Slide window: keep last (max_seq_len - 1) columns
                current = current[:, -(max_seq_len - 1):, :]

            out, _ = transformer(current)
            # Take the last predicted column as next input
            next_col = out[:, -1:, :]  # (1, 1, n_freq)
            future_cols.append(next_col.squeeze(0).cpu().numpy())
            current = torch.cat([current, next_col], dim=1)

    future_columns = np.concatenate(future_cols, axis=0) if future_cols else np.empty((0, n_freq))

    return {
        "injected_output": injected_output,
        "future_columns": future_columns,
        "original_output": original_output,
    }


def concept_scan(
    transformer: torch.nn.Module,
    vqvae: torch.nn.Module,
    input_spectrogram: torch.Tensor,
    scan_position: int,
    source_layer: int,
) -> np.ndarray:
    """Try every codebook entry at one position, collect output columns.

    Parameters
    ----------
    transformer:
        SpectrogramTransformer instance.
    vqvae:
        HiddenStateVQVAE instance.
    input_spectrogram:
        (1, S, n_freq) input spectrogram.
    scan_position:
        Position to scan.
    source_layer:
        1-indexed injection layer.

    Returns
    -------
    (K, n_freq) matrix: row k = output column at scan_position when
    codebook entry k is injected.
    """
    transformer.eval()
    vqvae.eval()
    K = vqvae.config.codebook_size
    results = []

    with torch.no_grad():
        for code_id in range(K):
            output = inject_and_continue(
                transformer, vqvae, input_spectrogram,
                scan_position, code_id, source_layer,
            )
            # Extract the column at scan_position
            results.append(output[0, scan_position].cpu().numpy())

    return np.stack(results)  # (K, n_freq)


def top_k_competing_concepts(
    vqvae: torch.nn.Module,
    hidden_states: torch.Tensor,
    k: int = 4,
) -> dict:
    """Find the top-k nearest codebook entries at each position.

    For each position, computes distances to all codebook entries and
    returns the k closest.  Useful for seeing which "concepts" compete
    at each time step.

    Parameters
    ----------
    vqvae:
        HiddenStateVQVAE instance.
    hidden_states:
        (1, S, d_model) hidden states.
    k:
        Number of top entries to return.

    Returns
    -------
    Dict with:
        indices: (S, k) top-k code indices per position
        distances: (S, k) distances to top-k codes
    """
    vqvae.eval()
    with torch.no_grad():
        # Encode and normalize
        z_e = vqvae.encoder(hidden_states)
        z_e = torch.nn.functional.normalize(z_e, dim=-1)
        z_flat = z_e.squeeze(0)  # (S, codebook_dim)

        # Distances to codebook
        codebook = vqvae.quantizer.embedding.weight  # (K, codebook_dim)
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ codebook.T
            + codebook.pow(2).sum(dim=1, keepdim=True).T
        )

        # Top-k (smallest distances)
        top_dists, top_indices = torch.topk(distances, k, dim=1, largest=False)

    return {
        "indices": top_indices.cpu().numpy(),
        "distances": top_dists.cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def plot_concept_scan(
    scan_result: np.ndarray,
    config: "AnalysisConfig",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot concept scan heatmap: codes x frequency bins.

    Parameters
    ----------
    scan_result:
        (K, n_freq) from ``concept_scan``.
    config:
        AnalysisConfig for axis labels.
    output_path:
        If provided, save figure.

    Returns
    -------
    matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(scan_result, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("Frequency bin")
    ax.set_ylabel("Codebook entry")
    ax.set_title("Concept Scan: Output Column per Codebook Entry")
    plt.colorbar(im, ax=ax, label="Amplitude")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=config.output_dpi, bbox_inches="tight")

    return fig


def plot_top_k_timeline(
    top_k_result: dict,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot timeline of competing concepts at each position.

    Parameters
    ----------
    top_k_result:
        Output from ``top_k_competing_concepts``.
    output_path:
        If provided, save figure.

    Returns
    -------
    matplotlib Figure.
    """
    indices = top_k_result["indices"]  # (S, k)
    S, k = indices.shape

    fig, ax = plt.subplots(figsize=(12, 4))
    for rank in range(k):
        ax.scatter(
            range(S), indices[:, rank],
            s=10, alpha=0.6, label=f"Rank {rank + 1}",
        )

    ax.set_xlabel("Position")
    ax.set_ylabel("Code ID")
    ax.set_title("Top-k Competing Concepts over Time")
    ax.legend(markerscale=3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_injection_comparison(
    result: dict,
    injection_position: int,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Side-by-side comparison of original vs injected spectrogram.

    Parameters
    ----------
    result:
        Output from ``concept_injection``.
    injection_position:
        Position where injection occurred (for marking).
    output_path:
        If provided, save figure.

    Returns
    -------
    matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    original = result["original_output"].T  # (n_freq, S)
    injected = result["injected_output"].T

    axes[0].imshow(original, aspect="auto", origin="lower", cmap="viridis")
    axes[0].axvline(injection_position, color="red", linewidth=1, linestyle="--")
    axes[0].set_title("Original")
    axes[0].set_ylabel("Freq bin")

    axes[1].imshow(injected, aspect="auto", origin="lower", cmap="viridis")
    axes[1].axvline(injection_position, color="red", linewidth=1, linestyle="--")
    axes[1].set_title("After Injection")

    diff = np.abs(injected - original)
    axes[2].imshow(diff, aspect="auto", origin="lower", cmap="hot")
    axes[2].axvline(injection_position, color="white", linewidth=1, linestyle="--")
    axes[2].set_title("Absolute Difference")

    for ax in axes:
        ax.set_xlabel("Time")

    fig.suptitle("Concept Injection Comparison")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
