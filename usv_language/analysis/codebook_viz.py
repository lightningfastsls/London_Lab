"""Codebook visualization: decode entries, exemplars, usage, projections.

For the two-phase architecture (ADR-007), decoding a codebook entry to a
spectrogram requires the VQ-VAE decoder (codebook_dim -> d_model) followed
by the transformer suffix (remaining layers -> output_head -> n_freq).
"""

from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from usv_language.analysis.config import AnalysisConfig
from usv_language.analysis.transformer_suffix import decode_hidden_to_spectrogram


def decode_all_entries(
    transformer: torch.nn.Module,
    vqvae: torch.nn.Module,
    source_layer: int,
) -> np.ndarray:
    """Decode every codebook entry to a spectrogram frequency profile.

    Path: codebook vector -> VQ-VAE decoder -> transformer suffix -> n_freq.

    Parameters
    ----------
    transformer:
        SpectrogramTransformer instance.
    vqvae:
        HiddenStateVQVAE instance.
    source_layer:
        1-indexed layer (hidden states captured here).

    Returns
    -------
    (K, n_freq) array of decoded frequency profiles.
    """
    transformer.eval()
    vqvae.eval()
    K = vqvae.config.codebook_size

    with torch.no_grad():
        # Get all codebook vectors: (K, codebook_dim)
        device = next(vqvae.parameters()).device
        indices = torch.arange(K, device=device)
        code_vectors = vqvae.quantizer.decode_indices(indices)

        # Decode to d_model space: (K, d_model)
        hidden = vqvae.decoder(code_vectors)

        # Decode each entry independently to avoid causal attention
        # cross-contamination.  Reshape as (K, 1, d_model) — each entry
        # is its own batch item with sequence length 1.
        hidden = hidden.unsqueeze(1)  # (K, 1, d_model)

        # Run through remaining transformer layers
        output = decode_hidden_to_spectrogram(
            transformer, hidden, start_layer=source_layer,
        )  # (K, 1, n_freq)

    return output.squeeze(1).cpu().numpy()  # (K, n_freq)


def find_exemplars(
    vqvae: torch.nn.Module,
    hidden_states: np.ndarray,
    n_exemplars: int = 10,
    batch_size: int = 4096,
) -> dict[int, np.ndarray]:
    """Find exemplar frame indices for each codebook entry.

    Processes hidden states in batches to avoid OOM, then collects
    the closest frames to each codebook entry.

    Parameters
    ----------
    vqvae:
        HiddenStateVQVAE instance.
    hidden_states:
        (N, d_model) array of hidden states.
    n_exemplars:
        Number of exemplar frames per code.
    batch_size:
        Processing batch size.

    Returns
    -------
    Dict mapping code_id -> array of frame indices (up to n_exemplars).
    """
    vqvae.eval()
    device = next(vqvae.parameters()).device
    N = hidden_states.shape[0]
    K = vqvae.config.codebook_size

    # Collect all code assignments
    all_codes = []
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            chunk = torch.from_numpy(
                hidden_states[start:end]
            ).float().unsqueeze(0).to(device)
            codes = vqvae.encode_to_codes(chunk)
            all_codes.append(codes.squeeze(0).cpu().numpy())

    all_codes = np.concatenate(all_codes)

    # Collect exemplar indices per code
    exemplars: dict[int, np.ndarray] = {}
    for k in range(K):
        mask = all_codes == k
        indices = np.where(mask)[0]
        if len(indices) > n_exemplars:
            # Take evenly spaced exemplars
            step = len(indices) // n_exemplars
            indices = indices[::step][:n_exemplars]
        exemplars[k] = indices

    return exemplars


def plot_decoded_profiles(
    profiles: np.ndarray,
    config: AnalysisConfig,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot decoded frequency profiles for all codebook entries.

    Parameters
    ----------
    profiles:
        (K, n_freq) decoded profiles from ``decode_all_entries``.
    config:
        AnalysisConfig for axis labels.
    output_path:
        If provided, save figure.

    Returns
    -------
    matplotlib Figure.
    """
    K, n_freq = profiles.shape
    freqs_khz = np.linspace(
        config.freq_min_hz / 1000, config.freq_max_hz / 1000, n_freq,
    )

    # Grid layout
    ncols = min(8, K)
    nrows = (K + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2 * nrows), sharey=True)
    axes = np.atleast_2d(axes)

    for i in range(nrows * ncols):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        if i < K:
            ax.plot(freqs_khz, profiles[i], linewidth=0.8)
            ax.set_title(f"Code {i}", fontsize=8)
            if col == 0:
                ax.set_ylabel("Amplitude", fontsize=7)
            if row == nrows - 1:
                ax.set_xlabel("kHz", fontsize=7)
            ax.tick_params(labelsize=6)
        else:
            ax.axis("off")

    fig.suptitle("Decoded Codebook Frequency Profiles", fontsize=12)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=config.output_dpi, bbox_inches="tight")

    return fig


def plot_codebook_usage(
    vqvae: torch.nn.Module,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot codebook usage histogram from EMA cluster sizes.

    Parameters
    ----------
    vqvae:
        HiddenStateVQVAE instance.
    output_path:
        If provided, save figure.

    Returns
    -------
    matplotlib Figure.
    """
    cluster_sizes = vqvae.quantizer.ema_cluster_size.cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    sorted_idx = np.argsort(cluster_sizes)[::-1]
    ax1.bar(range(len(cluster_sizes)), cluster_sizes[sorted_idx], width=1.0)
    ax1.set_xlabel("Code (sorted by usage)")
    ax1.set_ylabel("EMA cluster size")
    ax1.set_title("Codebook Usage (sorted)")

    ax2.hist(cluster_sizes, bins=30, edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("EMA cluster size")
    ax2.set_ylabel("Number of codes")
    ax2.set_title("Distribution of Code Usage")

    n_dead = (cluster_sizes < vqvae.config.dead_code_threshold).sum()
    n_total = len(cluster_sizes)
    fig.suptitle(
        f"Codebook Utilization: {n_total - n_dead}/{n_total} active "
        f"({(n_total - n_dead) / n_total:.0%})"
    )
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_codebook_projection(
    vqvae: torch.nn.Module,
    profiles: np.ndarray,
    config: AnalysisConfig,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """2D projection of codebook entries via t-SNE or UMAP.

    Parameters
    ----------
    vqvae:
        HiddenStateVQVAE instance (for codebook vectors).
    profiles:
        (K, n_freq) decoded profiles (for coloring by peak frequency).
    config:
        AnalysisConfig with projection parameters.
    output_path:
        If provided, save figure.

    Returns
    -------
    matplotlib Figure.
    """
    codebook = vqvae.quantizer.embedding.weight.detach().cpu().numpy()
    K = codebook.shape[0]

    # Try UMAP first, fall back to t-SNE
    coords_2d = None
    method_name = "t-SNE"
    try:
        import umap
        reducer = umap.UMAP(
            n_neighbors=min(config.umap_n_neighbors, K - 1),
            n_components=2,
            random_state=42,
        )
        coords_2d = reducer.fit_transform(codebook)
        method_name = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        tsne = TSNE(
            n_components=2,
            perplexity=min(config.tsne_perplexity, K - 1),
            random_state=42,
        )
        coords_2d = tsne.fit_transform(codebook)

    # Color by peak frequency
    peak_freq = np.argmax(profiles, axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        coords_2d[:, 0], coords_2d[:, 1],
        c=peak_freq, cmap="viridis", s=80, edgecolors="black", linewidth=0.5,
    )
    plt.colorbar(scatter, ax=ax, label="Peak frequency bin")

    for i in range(K):
        ax.annotate(str(i), (coords_2d[i, 0], coords_2d[i, 1]), fontsize=6, ha="center")

    ax.set_title(f"Codebook {method_name} Projection (colored by peak freq)")
    ax.set_xlabel(f"{method_name} dim 1")
    ax.set_ylabel(f"{method_name} dim 2")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=config.output_dpi, bbox_inches="tight")

    return fig


def plot_exemplar_gallery(
    code_id: int,
    exemplar_indices: np.ndarray,
    hidden_states: np.ndarray,
    transformer: torch.nn.Module,
    source_layer: int,
    config: AnalysisConfig,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot decoded spectrogram snippets around exemplar frames.

    Parameters
    ----------
    code_id:
        Codebook entry to visualize.
    exemplar_indices:
        Frame indices from ``find_exemplars``.
    hidden_states:
        (N, d_model) array.
    transformer:
        SpectrogramTransformer instance.
    source_layer:
        1-indexed layer.
    config:
        AnalysisConfig.
    output_path:
        If provided, save figure.

    Returns
    -------
    matplotlib Figure.
    """
    transformer.eval()
    n_show = min(len(exemplar_indices), 5)
    device = next(transformer.parameters()).device

    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 3))
    if n_show == 1:
        axes = [axes]

    for ax_idx, frame_idx in enumerate(exemplar_indices[:n_show]):
        # Extract context window around exemplar
        ctx = config.context_frames
        start = max(0, frame_idx - ctx // 2)
        end = min(len(hidden_states), start + ctx)
        snippet = hidden_states[start:end]

        # Decode through transformer suffix
        with torch.no_grad():
            h = torch.from_numpy(snippet).float().unsqueeze(0).to(device)
            spec = decode_hidden_to_spectrogram(transformer, h, source_layer)
            spec = spec.squeeze(0).cpu().numpy().T  # (n_freq, time)

        axes[ax_idx].imshow(spec, aspect="auto", origin="lower", cmap="viridis")
        axes[ax_idx].set_title(f"Frame {frame_idx}", fontsize=8)
        axes[ax_idx].set_xlabel("Time")
        if ax_idx == 0:
            axes[ax_idx].set_ylabel("Freq bin")

    fig.suptitle(f"Exemplars for Code {code_id}", fontsize=11)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=config.output_dpi, bbox_inches="tight")

    return fig
