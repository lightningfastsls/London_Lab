"""Codebook visualization: decode entries, plot frequency profiles, analyze usage.

Tools for understanding what each discrete code represents in terms of
frequency content, and how the codebook is utilized across the dataset.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from usv_language.src.model.vqvae import VQVAE


def decode_codebook_entries(model: VQVAE) -> np.ndarray:
    """Decode every codebook entry through the decoder to get frequency profiles.

    Parameters
    ----------
    model:
        Trained VQVAE model.

    Returns
    -------
    profiles: (codebook_size, n_freq) — decoded frequency profile for each code.
    """
    model.eval()
    K = model.config.codebook_size
    d = model.config.d_model

    with torch.no_grad():
        # Get all codebook vectors
        indices = torch.arange(K)
        vectors = model.quantizer.decode_indices(indices)  # (K, d_model)

        # Reshape as (1, K, d_model) batch for decoder
        vectors = vectors.unsqueeze(0)
        # Skip transformer for direct decode (shows raw code content)
        profiles = model.decoder(vectors)  # (1, K, n_freq)

    return profiles.squeeze(0).cpu().numpy()


def plot_codebook_profiles(
    profiles: np.ndarray,
    freq_min_hz: int = 20_000,
    freq_max_hz: int = 120_000,
    top_n: int = 20,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot frequency profiles of the most-used codebook entries.

    Parameters
    ----------
    profiles:
        (codebook_size, n_freq) decoded profiles.
    freq_min_hz, freq_max_hz:
        Frequency range for x-axis labels.
    top_n:
        Number of profiles to plot.
    output_path:
        If provided, save figure to this path.

    Returns
    -------
    matplotlib Figure.
    """
    n_freq = profiles.shape[1]
    freqs_khz = np.linspace(freq_min_hz / 1000, freq_max_hz / 1000, n_freq)

    fig, axes = plt.subplots(4, 5, figsize=(16, 12), sharey=True)
    for i, ax in enumerate(axes.ravel()):
        if i >= min(top_n, profiles.shape[0]):
            ax.axis("off")
            continue
        ax.plot(freqs_khz, profiles[i], linewidth=0.8)
        ax.set_title(f"Code {i}", fontsize=9)
        ax.set_xlabel("kHz", fontsize=7)
        if i % 5 == 0:
            ax.set_ylabel("Amplitude")
        ax.tick_params(labelsize=7)

    fig.suptitle("Codebook Entry Frequency Profiles")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_codebook_usage(
    model: VQVAE,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot codebook usage histogram from EMA cluster sizes.

    Parameters
    ----------
    model:
        Trained VQVAE model.
    output_path:
        If provided, save figure to this path.

    Returns
    -------
    matplotlib Figure.
    """
    cluster_sizes = model.quantizer.ema_cluster_size.cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Bar plot sorted by usage
    sorted_idx = np.argsort(cluster_sizes)[::-1]
    ax1.bar(range(len(cluster_sizes)), cluster_sizes[sorted_idx], width=1.0)
    ax1.set_xlabel("Code (sorted by usage)")
    ax1.set_ylabel("EMA cluster size")
    ax1.set_title("Codebook Usage (sorted)")

    # Histogram of usage counts
    ax2.hist(cluster_sizes, bins=50, edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("EMA cluster size")
    ax2.set_ylabel("Number of codes")
    ax2.set_title("Distribution of Code Usage")
    ax2.axvline(x=0.01, color="red", linestyle="--", label="Dead threshold")
    ax2.legend()

    n_dead = (cluster_sizes < 0.01).sum()
    n_total = len(cluster_sizes)
    fig.suptitle(f"Codebook Utilization: {n_total - n_dead}/{n_total} active codes ({(n_total - n_dead)/n_total:.0%})")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def cluster_codebook_entries(
    profiles: np.ndarray,
    n_clusters: int = 8,
) -> dict:
    """Cluster codebook entries by their decoded frequency profiles.

    Parameters
    ----------
    profiles:
        (codebook_size, n_freq) decoded profiles.
    n_clusters:
        Number of clusters.

    Returns
    -------
    Dict with "labels" (array), "centroids" (array), "inertia" (float).
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(profiles)
    return {
        "labels": labels,
        "centroids": km.cluster_centers_,
        "inertia": km.inertia_,
    }
