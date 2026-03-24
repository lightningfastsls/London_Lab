"""Compositionality tests: bigram productivity and positional independence.

These tests probe whether the codebook entries combine productively
(like words in a language) rather than as fixed sequences (like bird song).

Key metrics:
- Bigram productivity: what fraction of possible bigrams actually occur?
  Higher = more combinatorial freedom.
- Positional independence: are code meanings stable across positions?
  Low correlation between position and codebook distance = position-independent.
"""

from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from usv_language.analysis.transformer_suffix import decode_hidden_to_spectrogram


def bigram_productivity(codes: np.ndarray, K: int) -> dict:
    """Measure combinatorial freedom of the codebook.

    Counts how many distinct bigrams occur out of K^2 possible.
    Higher ratio suggests more productive combination.

    Parameters
    ----------
    codes:
        1D array of codebook indices.
    K:
        Codebook size.

    Returns
    -------
    Dict with ``observed`` (count), ``possible`` (K^2),
    ``productivity_ratio`` (observed/possible).
    """
    bigrams = set()
    for i in range(len(codes) - 1):
        bigrams.add((int(codes[i]), int(codes[i + 1])))

    possible = K * K
    observed = len(bigrams)

    return {
        "observed": observed,
        "possible": possible,
        "productivity_ratio": observed / possible if possible > 0 else 0.0,
    }


def heldout_bigram_decoding(
    codes: np.ndarray,
    K: int,
    transformer: torch.nn.Module,
    vqvae: torch.nn.Module,
    source_layer: int,
    n_heldout: int = 100,
) -> dict:
    """Test if novel bigrams produce coherent spectrograms.

    Finds bigrams that do NOT appear in the training codes,
    synthesizes them, and measures whether the decoded spectrogram
    is coherent (low variance, non-degenerate).

    Parameters
    ----------
    codes:
        1D array of codebook indices (training data).
    K:
        Codebook size.
    transformer:
        SpectrogramTransformer instance.
    vqvae:
        HiddenStateVQVAE instance.
    source_layer:
        1-indexed source layer.
    n_heldout:
        Maximum number of heldout bigrams to test.

    Returns
    -------
    Dict with ``n_heldout_tested``, ``mean_norm``, ``std_norm``,
    ``degenerate_fraction`` (norms near zero).
    """
    transformer.eval()
    vqvae.eval()
    device = next(transformer.parameters()).device

    # Find seen bigrams
    seen = set()
    for i in range(len(codes) - 1):
        seen.add((int(codes[i]), int(codes[i + 1])))

    # Find unseen bigrams
    unseen = []
    for a in range(K):
        for b in range(K):
            if (a, b) not in seen:
                unseen.append((a, b))
                if len(unseen) >= n_heldout:
                    break
        if len(unseen) >= n_heldout:
            break

    if not unseen:
        return {
            "n_heldout_tested": 0,
            "mean_norm": 0.0,
            "std_norm": 0.0,
            "degenerate_fraction": 0.0,
        }

    # Decode each unseen bigram
    norms = []
    with torch.no_grad():
        for a, b in unseen:
            idx = torch.tensor([a, b], device=device, dtype=torch.long)
            code_vecs = vqvae.quantizer.decode_indices(idx)  # (2, codebook_dim)
            hidden = vqvae.decoder(code_vecs).unsqueeze(0)  # (1, 2, d_model)
            spec = decode_hidden_to_spectrogram(
                transformer, hidden, start_layer=source_layer,
            )
            norm = spec.norm().item()
            norms.append(norm)

    norms = np.array(norms)
    degenerate = (norms < 1e-3).sum() / len(norms)

    return {
        "n_heldout_tested": len(unseen),
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std()),
        "degenerate_fraction": float(degenerate),
    }


def positional_independence(
    codes: np.ndarray,
    hidden_states: np.ndarray,
    vqvae: torch.nn.Module,
    K: int,
) -> dict:
    """Test whether code meaning depends on position.

    For each code k, computes the mean hidden state at each position
    where k appears, then measures Spearman correlation between position
    and distance from the code's global centroid.

    Low |correlation| = position-independent meaning (language-like).
    High |correlation| = position-dependent (more like fixed templates).

    Parameters
    ----------
    codes:
        1D array of codebook indices.
    hidden_states:
        (N, d_model) hidden states matching ``codes``.
    vqvae:
        HiddenStateVQVAE instance.
    K:
        Codebook size.

    Returns
    -------
    Dict with ``mean_abs_correlation``, ``per_code_correlations``,
    ``codes_tested`` (number of codes with enough data).
    """
    from scipy.stats import spearmanr

    correlations: dict[int, float] = {}

    for k in range(K):
        positions = np.where(codes == k)[0]
        if len(positions) < 10:
            continue

        # Hidden states at these positions
        hs = hidden_states[positions]  # (n_occurrences, d_model)
        centroid = hs.mean(axis=0)  # (d_model,)

        # Distance from centroid at each occurrence
        dists = np.linalg.norm(hs - centroid, axis=1)

        # Spearman correlation between position and distance
        rho, _ = spearmanr(positions, dists)
        if not np.isnan(rho):
            correlations[k] = float(rho)

    if not correlations:
        return {
            "mean_abs_correlation": 0.0,
            "per_code_correlations": {},
            "codes_tested": 0,
        }

    abs_corrs = [abs(v) for v in correlations.values()]
    return {
        "mean_abs_correlation": float(np.mean(abs_corrs)),
        "per_code_correlations": correlations,
        "codes_tested": len(correlations),
    }


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def plot_bigram_productivity(
    productivity_result: dict,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize bigram productivity as bar chart.

    Parameters
    ----------
    productivity_result:
        Output from ``bigram_productivity``.
    output_path:
        If provided, save figure.

    Returns
    -------
    matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    observed = productivity_result["observed"]
    possible = productivity_result["possible"]
    ratio = productivity_result["productivity_ratio"]

    ax.bar(["Observed", "Possible"], [observed, possible], color=["steelblue", "lightgray"])
    ax.set_ylabel("Number of bigrams")
    ax.set_title(f"Bigram Productivity: {ratio:.1%} ({observed}/{possible})")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_positional_independence(
    pos_result: dict,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot per-code positional independence correlations.

    Parameters
    ----------
    pos_result:
        Output from ``positional_independence``.
    output_path:
        If provided, save figure.

    Returns
    -------
    matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    corrs = pos_result["per_code_correlations"]
    if not corrs:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        return fig

    code_ids = sorted(corrs.keys())
    values = [corrs[k] for k in code_ids]

    ax.bar(code_ids, values, width=0.8, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Code ID")
    ax.set_ylabel("Spearman correlation (position vs distance)")
    mean_abs = pos_result["mean_abs_correlation"]
    ax.set_title(f"Positional Independence (mean |rho| = {mean_abs:.3f})")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
