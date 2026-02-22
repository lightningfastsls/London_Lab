"""Sequence analysis: transition matrices, n-grams, mutual information, entropy.

Analyzes the sequential structure of codebook indices to discover
grammar-like patterns in USV vocalizations.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def compute_transition_matrix(
    codes: np.ndarray,
    codebook_size: int,
) -> np.ndarray:
    """Compute code-to-code transition probability matrix.

    Parameters
    ----------
    codes:
        1D array of codebook indices.
    codebook_size:
        Total number of codes (K).

    Returns
    -------
    P: (K, K) transition probability matrix where P[i,j] = P(code_j | code_i).
    """
    counts = np.zeros((codebook_size, codebook_size), dtype=np.float64)
    for i in range(len(codes) - 1):
        counts[codes[i], codes[i + 1]] += 1

    # Normalize rows
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    return counts / row_sums


def transition_entropy(
    transition_matrix: np.ndarray,
) -> np.ndarray:
    """Compute entropy of transitions from each code.

    High entropy = many possible continuations (unpredictable).
    Low entropy = few likely continuations (predictable, structural).

    Parameters
    ----------
    transition_matrix:
        (K, K) probability matrix.

    Returns
    -------
    (K,) entropy values per code.
    """
    eps = 1e-12
    P = transition_matrix + eps
    return -np.sum(P * np.log2(P), axis=1)


def extract_ngrams(
    codes: np.ndarray,
    n: int,
) -> Counter:
    """Extract n-grams from a code sequence.

    Parameters
    ----------
    codes:
        1D array of codebook indices.
    n:
        N-gram order.

    Returns
    -------
    Counter of n-gram tuples.
    """
    ngrams = Counter()
    for i in range(len(codes) - n + 1):
        ngram = tuple(codes[i:i + n].tolist())
        ngrams[ngram] += 1
    return ngrams


def top_ngrams(
    codes: np.ndarray,
    n: int,
    top_k: int = 20,
) -> List[Tuple[tuple, int]]:
    """Get the top-K most frequent n-grams.

    Parameters
    ----------
    codes: 1D array of codebook indices.
    n: N-gram order.
    top_k: Number of top n-grams to return.

    Returns
    -------
    List of (ngram_tuple, count) pairs sorted by frequency.
    """
    ngrams = extract_ngrams(codes, n)
    return ngrams.most_common(top_k)


def mutual_information_at_lag(
    codes: np.ndarray,
    codebook_size: int,
    lag: int,
) -> float:
    """Compute mutual information between code at position t and t+lag.

    MI(X_t; X_{t+lag}) = sum_ij P(i,j) * log2(P(i,j) / (P(i)*P(j)))

    Parameters
    ----------
    codes: 1D array of codebook indices.
    codebook_size: Total number of codes.
    lag: Temporal lag.

    Returns
    -------
    Mutual information in bits.
    """
    if lag >= len(codes):
        return 0.0

    joint = np.zeros((codebook_size, codebook_size), dtype=np.float64)
    for i in range(len(codes) - lag):
        joint[codes[i], codes[i + lag]] += 1

    total = joint.sum()
    if total == 0:
        return 0.0

    joint /= total
    marginal_x = joint.sum(axis=1)
    marginal_y = joint.sum(axis=0)

    mi = 0.0
    for i in range(codebook_size):
        for j in range(codebook_size):
            if joint[i, j] > 0 and marginal_x[i] > 0 and marginal_y[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (marginal_x[i] * marginal_y[j]))
    return mi


def entropy_rate(
    codes: np.ndarray,
    max_order: int = 5,
) -> List[float]:
    """Compute entropy rate at different n-gram orders.

    Decreasing entropy rate with order indicates sequential structure.

    Parameters
    ----------
    codes: 1D array of codebook indices.
    max_order: Maximum n-gram order to compute.

    Returns
    -------
    List of entropy rates for orders 1 through max_order.
    """
    rates = []
    for n in range(1, max_order + 1):
        ngrams = extract_ngrams(codes, n)
        total = sum(ngrams.values())
        if total == 0:
            rates.append(0.0)
            continue

        entropy = 0.0
        for count in ngrams.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)

        # Entropy rate = H(n-gram) / n
        rates.append(entropy / n)
    return rates


def plot_transition_matrix(
    transition_matrix: np.ndarray,
    top_n: int = 30,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot transition matrix heatmap for the most-used codes.

    Parameters
    ----------
    transition_matrix: (K, K) probability matrix.
    top_n: Number of codes to show.
    output_path: If provided, save figure.

    Returns
    -------
    matplotlib Figure.
    """
    # Select top-N most active codes (highest row sums in raw counts)
    activity = transition_matrix.sum(axis=1)
    top_indices = np.argsort(activity)[::-1][:top_n]

    sub_matrix = transition_matrix[np.ix_(top_indices, top_indices)]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sub_matrix, cmap="viridis", aspect="auto")
    ax.set_xlabel("Next code")
    ax.set_ylabel("Current code")
    ax.set_title(f"Transition Probabilities (top {top_n} codes)")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(top_indices, fontsize=6, rotation=90)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_indices, fontsize=6)
    plt.colorbar(im, ax=ax, label="P(next | current)")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
