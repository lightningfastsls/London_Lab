"""Sequential structure analysis: Zipf, transitions, entropy, mutual information.

Ports core algorithms from ``usv_language.src.analysis.sequence_analysis``
(Phase 7) and adds new functions for the two-phase architecture (Phase 8.4).

Key additions over Phase 7:
- ``extract_code_sequences``: batch-wise code extraction from hidden states
- ``extract_bout_code_sequences``: per-recording code extraction
- ``zipf_analysis``: OLS power-law fit on log-log code frequencies
- ``conditional_entropy``: H(C_{t+1} | C_t)
- ``mutual_information_bigram``: I(C_t; C_{t+1})
- ``excess_entropy``: I(past; future) with Laplace smoothing
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Ported from usv_language.src.analysis.sequence_analysis (same algorithms)
# ---------------------------------------------------------------------------


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
    (K, K) transition probability matrix where P[i,j] = P(code_j | code_i).
    """
    counts = np.zeros((codebook_size, codebook_size), dtype=np.float64)
    for i in range(len(codes) - 1):
        counts[codes[i], codes[i + 1]] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    return counts / row_sums


def transition_entropy(transition_matrix: np.ndarray) -> np.ndarray:
    """Compute entropy of transitions from each code.

    Parameters
    ----------
    transition_matrix:
        (K, K) probability matrix.

    Returns
    -------
    (K,) entropy values per code (bits).
    """
    eps = 1e-12
    P = transition_matrix + eps
    return -np.sum(P * np.log2(P), axis=1)


def extract_ngrams(codes: np.ndarray, n: int) -> Counter:
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
    ngrams: Counter = Counter()
    for i in range(len(codes) - n + 1):
        ngram = tuple(codes[i : i + n].tolist())
        ngrams[ngram] += 1
    return ngrams


def top_ngrams(
    codes: np.ndarray, n: int, top_k: int = 20,
) -> list[tuple[tuple, int]]:
    """Get the top-K most frequent n-grams."""
    return extract_ngrams(codes, n).most_common(top_k)


def mutual_information_at_lag(
    codes: np.ndarray,
    codebook_size: int,
    lag: int,
) -> float:
    """Compute mutual information between code at position t and t+lag.

    MI(X_t; X_{t+lag}) = sum_ij P(i,j) * log2(P(i,j) / (P(i)*P(j)))

    Parameters
    ----------
    codes:
        1D array of codebook indices.
    codebook_size:
        Total number of codes (K).
    lag:
        Temporal lag.

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
                mi += joint[i, j] * np.log2(
                    joint[i, j] / (marginal_x[i] * marginal_y[j])
                )
    return mi


def segment_into_bouts(
    codes: np.ndarray,
    ici_gap_s: np.ndarray,
    bout_threshold_s: float,
) -> list[np.ndarray]:
    """Split a flat code sequence into per-bout subsequences.

    A "bout" is a silence-bounded run of calls: two consecutive calls belong
    to the same bout iff the silent gap between them (``ici_gap_s[i]``, the
    end-to-start interval) is strictly less than ``bout_threshold_s``.
    Matches Phase A2 (``scripts/analyze_sequential_structure.py``): the
    threshold is an **upper bound** on within-bout gaps — a gap of exactly
    ``bout_threshold_s`` starts a new bout, a gap smaller than it does not.

    Parameters
    ----------
    codes:
        1D length-N array of integer codes.
    ici_gap_s:
        1D length-(N-1) array of silent gaps in seconds. Element ``i`` is
        the gap between call ``i`` and call ``i+1``.
    bout_threshold_s:
        Silent-gap threshold in seconds. Must be finite and non-negative.

    Returns
    -------
    list[np.ndarray]
        One array per bout. Includes singleton bouts (length 1) — the
        caller is responsible for dropping them when only pairs are needed
        (e.g. ``seq for seq in bouts if len(seq) >= 2``).

    Raises
    ------
    ValueError
        If ``len(ici_gap_s) != len(codes) - 1`` or ``bout_threshold_s`` is
        not finite / not non-negative.
    """
    codes = np.asarray(codes)
    gaps = np.asarray(ici_gap_s)
    n = int(codes.shape[0])
    if n == 0:
        return []
    if gaps.shape[0] != n - 1:
        raise ValueError(
            f"ici_gap_s length ({gaps.shape[0]}) must equal "
            f"len(codes) - 1 = {n - 1}"
        )
    if not np.isfinite(bout_threshold_s) or bout_threshold_s < 0:
        raise ValueError(
            f"bout_threshold_s must be finite and non-negative, "
            f"got {bout_threshold_s!r}"
        )
    if n == 1:
        return [codes.copy()]

    # Boundary indices (start of each new bout, after the first) are the
    # positions i+1 where gaps[i] > threshold. Strict '>' matches Phase
    # A2's detect_bouts (``if ici > threshold_s: bout_id += 1``) — a gap
    # equal to the threshold stays within the current bout.
    boundaries = np.where(gaps > bout_threshold_s)[0] + 1
    boundaries = np.concatenate([[0], boundaries, [n]])
    return [codes[boundaries[k]:boundaries[k + 1]] for k in range(len(boundaries) - 1)]


def mutual_information_from_sequences(
    sequences: list[np.ndarray],
    codebook_size: int,
    lag: int = 1,
) -> tuple[float, int]:
    """MI at lag ``lag`` counting only within-sequence pairs.

    Ignores cross-sequence transitions entirely. Intended as the shared
    primitive behind bout-aware analyses: callers first segment a flat
    sequence into bouts (e.g. via :func:`segment_into_bouts`) then pass
    the bout list here.

    Parameters
    ----------
    sequences:
        List of 1D integer code arrays. Singletons (length < 2) contribute
        no pairs and are skipped; they do not raise.
    codebook_size:
        Number of codes (K). The joint-count matrix is K×K.
    lag:
        Temporal lag between the two variables. Default 1.

    Returns
    -------
    tuple[float, int]
        (MI in bits, number of (T, T+lag) pairs counted).
        Returns (0.0, 0) when no pairs are available.
    """
    joint = np.zeros((codebook_size, codebook_size), dtype=np.float64)
    n_pairs = 0
    for seq in sequences:
        if len(seq) <= lag:
            continue
        for i in range(len(seq) - lag):
            joint[seq[i], seq[i + lag]] += 1
        n_pairs += len(seq) - lag

    total = joint.sum()
    if total == 0:
        return 0.0, 0

    joint /= total
    marginal_x = joint.sum(axis=1)
    marginal_y = joint.sum(axis=0)

    mi = 0.0
    for i in range(codebook_size):
        for j in range(codebook_size):
            if joint[i, j] > 0 and marginal_x[i] > 0 and marginal_y[j] > 0:
                mi += joint[i, j] * np.log2(
                    joint[i, j] / (marginal_x[i] * marginal_y[j])
                )
    return float(mi), int(n_pairs)


def mutual_information_within_bouts(
    codes: np.ndarray,
    ici_gap_s: np.ndarray,
    bout_threshold_s: float,
    codebook_size: int,
    lag: int = 1,
) -> tuple[float, int, int]:
    """Bout-filtered MI at ``lag`` on a flat code sequence.

    Convenience wrapper: segments ``codes`` by ``ici_gap_s`` at the given
    threshold, then computes MI on within-bout pairs only. Used by both
    Phase A2 (``scripts/analyze_sequential_structure.py``) and SIS 17.1
    (``scripts/run_sis_baselines.py``) — any refinement of bout filtering
    should happen here, so both consumers update together.

    Parameters
    ----------
    codes:
        1D length-N integer code array (already chronologically sorted).
    ici_gap_s:
        1D length-(N-1) silent-gap array (seconds, end-to-start).
    bout_threshold_s:
        Silent-gap threshold; pairs with gap ``>=`` threshold are treated
        as cross-bout (excluded). Matches Phase A2's convention.
    codebook_size:
        Number of codes (K) in the alphabet.
    lag:
        Temporal lag between the two variables. Default 1.

    Returns
    -------
    tuple[float, int, int]
        (MI in bits, n_within_pairs, n_excluded_pairs). The two counts sum
        to ``len(codes) - lag`` when ``lag`` == 1 and every cross-bout
        transition contributes one excluded pair; at higher lags a pair is
        excluded if its two endpoints are in different bouts.
    """
    bouts = segment_into_bouts(codes, ici_gap_s, bout_threshold_s)
    mi, n_within = mutual_information_from_sequences(bouts, codebook_size, lag=lag)
    n_total_pairs = max(0, len(codes) - lag)
    n_excluded = n_total_pairs - n_within
    return mi, n_within, n_excluded


def entropy_rate(codes: np.ndarray, max_order: int = 5) -> list[float]:
    """Compute entropy rate at different n-gram orders.

    Decreasing entropy rate with order indicates sequential structure.

    Parameters
    ----------
    codes:
        1D array of codebook indices.
    max_order:
        Maximum n-gram order to compute.

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

        ent = 0.0
        for count in ngrams.values():
            p = count / total
            if p > 0:
                ent -= p * np.log2(p)

        rates.append(ent / n)
    return rates


# ---------------------------------------------------------------------------
# New functions for two-phase architecture
# ---------------------------------------------------------------------------


def recording_spans_from_metadata(
    metadata: dict,
    total_frames: int | None = None,
) -> list[dict]:
    """Extract recording spans from metadata in new or legacy formats.

    Supports either:
    - ``recordings``: list of dicts with ``recording_id``, ``start_frame``,
      ``end_frame``
    - ``frames``: legacy per-frame entries with ``recording_id`` and
      ``frame_index``
    """
    if "recordings" in metadata and metadata["recordings"]:
        spans = []
        for recording in metadata["recordings"]:
            start = int(recording.get("start_frame", 0))
            end_default = total_frames if total_frames is not None else start
            end = int(recording.get("end_frame", end_default))
            if total_frames is not None:
                end = min(end, total_frames)
            if start < end:
                spans.append({
                    "recording_id": str(recording.get("recording_id", "unknown")),
                    "start_frame": start,
                    "end_frame": end,
                })
        return spans

    frames = metadata.get("frames", [])
    if not frames:
        return []

    ordered_frames = sorted(
        frames, key=lambda entry: int(entry.get("frame_index", 0))
    )
    spans: list[dict] = []
    current_recording = str(ordered_frames[0].get("recording_id", "unknown"))
    start_frame = int(ordered_frames[0].get("frame_index", 0))
    previous_frame = start_frame

    for entry in ordered_frames[1:]:
        recording_id = str(entry.get("recording_id", "unknown"))
        frame_index = int(entry.get("frame_index", previous_frame + 1))
        contiguous = recording_id == current_recording and frame_index == previous_frame + 1
        if not contiguous:
            spans.append({
                "recording_id": current_recording,
                "start_frame": start_frame,
                "end_frame": previous_frame + 1,
            })
            current_recording = recording_id
            start_frame = frame_index
        previous_frame = frame_index

    spans.append({
        "recording_id": current_recording,
        "start_frame": start_frame,
        "end_frame": previous_frame + 1,
    })
    return spans


def extract_code_sequences(
    hidden_states: np.ndarray,
    vqvae: torch.nn.Module,
    batch_size: int = 4096,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Encode hidden states into discrete codes in batches.

    Parameters
    ----------
    hidden_states:
        (N, d_model) array of hidden states.
    vqvae:
        HiddenStateVQVAE model (uses ``encode_to_codes``).
    batch_size:
        Processing batch size to avoid OOM.
    device:
        Device for inference.

    Returns
    -------
    (N,) array of codebook indices.
    """
    vqvae.eval()
    N = hidden_states.shape[0]
    all_codes = []

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            chunk = torch.from_numpy(
                hidden_states[start:end]
            ).float().unsqueeze(0).to(device)
            codes = vqvae.encode_to_codes(chunk)  # (1, chunk_len)
            all_codes.append(codes.squeeze(0).cpu().numpy())

    return np.concatenate(all_codes)


def extract_bout_code_sequences(
    hidden_states: np.ndarray,
    metadata_path: str | Path,
    vqvae: torch.nn.Module,
    batch_size: int = 4096,
    device: str | torch.device = "cpu",
) -> list[np.ndarray]:
    """Extract per-recording code sequences using metadata boundaries.

    Parameters
    ----------
    hidden_states:
        (N, d_model) array of hidden states.
    metadata_path:
        Path to metadata.json with ``recording_id`` and frame boundaries.
    vqvae:
        HiddenStateVQVAE model.
    batch_size:
        Processing batch size.
    device:
        Device for inference.

    Returns
    -------
    List of 1D code arrays, one per recording.
    """
    all_codes = extract_code_sequences(hidden_states, vqvae, batch_size, device)

    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        # No metadata — return single sequence
        return [all_codes]

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    spans = recording_spans_from_metadata(metadata, total_frames=len(all_codes))
    if not spans:
        return [all_codes]

    bout_sequences = []
    for recording in spans:
        start = recording["start_frame"]
        end = recording["end_frame"]
        if start < end:
            bout_sequences.append(all_codes[start:end])

    return bout_sequences if bout_sequences else [all_codes]


def zipf_analysis(codes: np.ndarray, min_count: int = 5) -> dict:
    """Fit Zipf's law to code frequency distribution.

    Performs OLS regression on log(rank) vs log(frequency).
    True Zipf's law gives slope (alpha) near -1.0.

    Parameters
    ----------
    codes:
        1D array of codebook indices.
    min_count:
        Minimum occurrence count to include in fit.

    Returns
    -------
    Dict with ``alpha`` (slope), ``r_squared``, ``ranks``, ``frequencies``.
    """
    counts = Counter(codes.tolist())
    freqs = sorted(counts.values(), reverse=True)
    freqs = [f for f in freqs if f >= min_count]

    if len(freqs) < 3:
        return {"alpha": 0.0, "r_squared": 0.0, "ranks": [], "frequencies": []}

    ranks = np.arange(1, len(freqs) + 1, dtype=np.float64)
    freqs_arr = np.array(freqs, dtype=np.float64)

    log_ranks = np.log10(ranks)
    log_freqs = np.log10(freqs_arr)

    # OLS: log_freq = alpha * log_rank + intercept
    A = np.vstack([log_ranks, np.ones(len(log_ranks))]).T
    result = np.linalg.lstsq(A, log_freqs, rcond=None)
    alpha, intercept = result[0]

    # R-squared
    predicted = alpha * log_ranks + intercept
    ss_res = np.sum((log_freqs - predicted) ** 2)
    ss_tot = np.sum((log_freqs - log_freqs.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "alpha": float(alpha),
        "r_squared": float(r_squared),
        "ranks": ranks.tolist(),
        "frequencies": freqs,
    }


def conditional_entropy(codes: np.ndarray, K: int) -> float:
    """Compute conditional entropy H(C_{t+1} | C_t).

    Weighted average of per-row entropies in the transition matrix.

    Parameters
    ----------
    codes:
        1D array of codebook indices.
    K:
        Codebook size.

    Returns
    -------
    Conditional entropy in bits.
    """
    trans = compute_transition_matrix(codes, K)

    # Marginal: fraction of transitions starting from each code
    counts = np.zeros(K, dtype=np.float64)
    for i in range(len(codes) - 1):
        counts[codes[i]] += 1
    total = counts.sum()
    if total == 0:
        return 0.0
    marginal = counts / total

    # Weighted row entropy
    row_ent = transition_entropy(trans)
    return float(np.sum(marginal * row_ent))


def mutual_information_bigram(codes: np.ndarray, K: int) -> float:
    """Compute mutual information I(C_t; C_{t+1}).

    Equivalent to H(C_{t+1}) - H(C_{t+1} | C_t).

    Parameters
    ----------
    codes:
        1D array of codebook indices.
    K:
        Codebook size.

    Returns
    -------
    Mutual information in bits.
    """
    return mutual_information_at_lag(codes, K, lag=1)


def excess_entropy(codes: np.ndarray, K: int, max_order: int = 8) -> float:
    """Estimate excess entropy E = I(past; future) via entropy rate convergence.

    Excess entropy is defined as the total mutual information between the
    semi-infinite past and future of a stationary process.  It can be
    computed from n-gram entropy rates as:

        E = sum_{n=1}^{L} (h_n - h_L)

    where h_n = H(n-gram)/n is the entropy rate at order n and h_L is the
    rate at the highest order (our best estimate of the true rate h).

    Higher E indicates more complex long-range dependencies — knowing the
    past reduces uncertainty about the future substantially.

    Parameters
    ----------
    codes:
        1D array of codebook indices.
    K:
        Codebook size.
    max_order:
        Maximum n-gram order for entropy rate estimation.

    Returns
    -------
    Excess entropy estimate in bits.
    """
    if len(codes) < 2:
        return 0.0

    rates = entropy_rate(codes, max_order=max_order)
    if not rates:
        return 0.0

    # h_L: best estimate of true entropy rate (highest order computed)
    h_inf = rates[-1]

    # E = sum of (h_n - h_inf) for n = 1..L-1
    excess = sum(r - h_inf for r in rates[:-1])
    return max(0.0, float(excess))


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def plot_zipf(
    zipf_result: dict,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Zipf's law: log(rank) vs log(frequency).

    Parameters
    ----------
    zipf_result:
        Output from ``zipf_analysis``.
    output_path:
        If provided, save figure.

    Returns
    -------
    matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if not zipf_result["ranks"]:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        return fig

    ranks = np.array(zipf_result["ranks"])
    freqs = np.array(zipf_result["frequencies"])

    ax.scatter(np.log10(ranks), np.log10(freqs), s=20, alpha=0.7, label="Data")

    # Fit line using OLS intercept
    alpha = zipf_result["alpha"]
    r2 = zipf_result["r_squared"]
    log_ranks = np.log10(ranks)
    log_freqs = np.log10(freqs)
    # Recompute intercept from OLS (same fit as zipf_analysis)
    A = np.vstack([log_ranks, np.ones(len(log_ranks))]).T
    coeffs = np.linalg.lstsq(A, log_freqs, rcond=None)[0]
    intercept = coeffs[1]
    ax.plot(
        log_ranks,
        alpha * log_ranks + intercept,
        "r--",
        label=f"Fit: alpha={alpha:.2f}, R²={r2:.3f}",
    )

    ax.set_xlabel("log10(rank)")
    ax.set_ylabel("log10(frequency)")
    ax.set_title("Zipf's Law Analysis")
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_entropy_rate(
    rates: list[float],
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot entropy rate vs n-gram order.

    Parameters
    ----------
    rates:
        Output from ``entropy_rate``.
    output_path:
        If provided, save figure.

    Returns
    -------
    matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    orders = list(range(1, len(rates) + 1))
    ax.plot(orders, rates, "bo-", linewidth=2, markersize=8)
    ax.set_xlabel("N-gram order")
    ax.set_ylabel("Entropy rate (bits/symbol)")
    ax.set_title("Entropy Rate by N-gram Order")
    ax.set_xticks(orders)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_mi_decay(
    codes: np.ndarray,
    codebook_size: int,
    max_lag: int = 20,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot mutual information decay over temporal lags.

    Parameters
    ----------
    codes:
        1D array of codebook indices.
    codebook_size:
        Total number of codes (K).
    max_lag:
        Maximum lag to compute.
    output_path:
        If provided, save figure.

    Returns
    -------
    matplotlib Figure.
    """
    lags = list(range(1, max_lag + 1))
    mi_values = [mutual_information_at_lag(codes, codebook_size, lag) for lag in lags]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lags, mi_values, "ro-", linewidth=2, markersize=6)
    ax.set_xlabel("Lag (time steps)")
    ax.set_ylabel("Mutual Information (bits)")
    ax.set_title("MI Decay over Temporal Lag")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_transition_matrix(
    transition_matrix: np.ndarray,
    top_n: int = 30,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot transition matrix heatmap for the most-used codes.

    Parameters
    ----------
    transition_matrix:
        (K, K) probability matrix.
    top_n:
        Number of codes to show.
    output_path:
        If provided, save figure.

    Returns
    -------
    matplotlib Figure.
    """
    activity = transition_matrix.sum(axis=1)
    top_n = min(top_n, len(activity))
    top_indices = np.argsort(activity)[::-1][:top_n]

    sub_matrix = transition_matrix[np.ix_(top_indices, top_indices)]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sub_matrix, cmap="viridis", aspect="auto")
    ax.set_xlabel("Next code")
    ax.set_ylabel("Current code")
    ax.set_title(f"Transition Probabilities (top {top_n} codes)")
    ax.set_xticks(range(len(top_indices)))
    ax.set_xticklabels(top_indices, fontsize=6, rotation=90)
    ax.set_yticks(range(len(top_indices)))
    ax.set_yticklabels(top_indices, fontsize=6)
    plt.colorbar(im, ax=ax, label="P(next | current)")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
