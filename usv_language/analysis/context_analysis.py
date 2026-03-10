"""Context-dependent analysis: compare code usage across groups.

Groups hidden-state-derived codes by metadata (recording_id, and
eventually sex, strain, social context) and computes statistical
differences between groups.

Note: current metadata only provides recording_id.  Functions gracefully
report when richer metadata (sex, strain, social context) is unavailable.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from usv_language.analysis.sequence_analysis import (
    compute_transition_matrix,
    recording_spans_from_metadata,
)


def group_codes_by_metadata(
    codes: np.ndarray,
    metadata: dict,
    group_key: str = "recording_id",
) -> dict[str, np.ndarray]:
    """Split code sequence into groups based on metadata.

    Parameters
    ----------
    codes:
        1D array of codebook indices.
    metadata:
        Metadata dict, expected to have ``recordings`` list with
        ``recording_id``, ``start_frame``, ``end_frame``.
    group_key:
        Key to group by (default ``recording_id``).

    Returns
    -------
    Dict mapping group label -> 1D code array.
    """
    spans = recording_spans_from_metadata(metadata, total_frames=len(codes))
    if not spans:
        return {"all": codes}

    grouped_lists: dict[str, list[np.ndarray]] = {}
    for recording in spans:
        label = str(recording.get(group_key, recording.get("recording_id", "unknown")))
        start = recording["start_frame"]
        end = recording["end_frame"]
        if start < end:
            grouped_lists.setdefault(label, []).append(codes[start:end])

    if not grouped_lists:
        return {"all": codes}

    return {
        label: np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        for label, chunks in grouped_lists.items()
    }


def code_frequency_by_group(
    grouped_codes: dict[str, np.ndarray],
    K: int,
) -> dict[str, np.ndarray]:
    """Compute code frequency distributions per group.

    Parameters
    ----------
    grouped_codes:
        Dict mapping group label -> 1D code array.
    K:
        Codebook size.

    Returns
    -------
    Dict mapping group label -> (K,) normalized frequency array.
    """
    result = {}
    for label, codes in grouped_codes.items():
        counts = np.bincount(codes.astype(int), minlength=K).astype(np.float64)
        total = counts.sum()
        result[label] = counts / total if total > 0 else counts
    return result


def chi_squared_test(
    freq_a: np.ndarray,
    freq_b: np.ndarray,
) -> dict:
    """Chi-squared test for difference between two code distributions.

    Parameters
    ----------
    freq_a:
        (K,) normalized frequency distribution A.
    freq_b:
        (K,) normalized frequency distribution B.

    Returns
    -------
    Dict with ``chi2`` statistic, ``p_value``, ``df`` degrees of freedom.
    """
    # Convert to counts (use 1000 as reference total for stable test)
    n = 1000
    obs_a = (freq_a * n).astype(np.float64)
    obs_b = (freq_b * n).astype(np.float64)

    # Filter out bins where both are zero
    mask = (obs_a + obs_b) > 0
    if mask.sum() < 2:
        return {"chi2": 0.0, "p_value": 1.0, "df": 0}

    observed = np.array([obs_a[mask], obs_b[mask]])
    chi2, p_value, df, _ = stats.chi2_contingency(observed)

    return {"chi2": float(chi2), "p_value": float(p_value), "df": int(df)}


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence D_KL(P || Q) with smoothing.

    Parameters
    ----------
    p:
        (K,) distribution P.
    q:
        (K,) distribution Q.

    Returns
    -------
    KL divergence in nats.
    """
    eps = 1e-10
    p_smooth = p + eps
    q_smooth = q + eps
    p_smooth = p_smooth / p_smooth.sum()
    q_smooth = q_smooth / q_smooth.sum()
    return float(np.sum(p_smooth * np.log(p_smooth / q_smooth)))


def differential_code_usage(
    grouped_codes: dict[str, np.ndarray],
    K: int,
) -> list[dict]:
    """Find codes with significantly different usage across groups.

    Compares each pair of groups using chi-squared tests.

    Parameters
    ----------
    grouped_codes:
        Dict mapping group label -> 1D code array.
    K:
        Codebook size.

    Returns
    -------
    List of dicts with ``group_a``, ``group_b``, ``chi2``, ``p_value``,
    ``kl_ab``, ``kl_ba``, ``top_differential_codes``.
    """
    freqs = code_frequency_by_group(grouped_codes, K)
    labels = sorted(freqs.keys())
    results = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            chi = chi_squared_test(freqs[a], freqs[b])
            kl_ab = kl_divergence(freqs[a], freqs[b])
            kl_ba = kl_divergence(freqs[b], freqs[a])

            # Find codes with largest frequency difference
            diff = np.abs(freqs[a] - freqs[b])
            top_codes = np.argsort(diff)[::-1][:5].tolist()

            results.append({
                "group_a": a,
                "group_b": b,
                "chi2": chi["chi2"],
                "p_value": chi["p_value"],
                "kl_ab": kl_ab,
                "kl_ba": kl_ba,
                "top_differential_codes": top_codes,
            })

    return results


def compare_transition_matrices(
    grouped_codes: dict[str, np.ndarray],
    K: int,
) -> dict:
    """Compare transition matrices across groups.

    Parameters
    ----------
    grouped_codes:
        Dict mapping group label -> 1D code array.
    K:
        Codebook size.

    Returns
    -------
    Dict with ``matrices`` (per-group), ``frobenius_distances`` (pairwise).
    """
    matrices = {}
    for label, codes in grouped_codes.items():
        matrices[label] = compute_transition_matrix(codes, K)

    labels = sorted(matrices.keys())
    frobenius = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            dist = np.linalg.norm(matrices[a] - matrices[b], "fro")
            frobenius[f"{a}_vs_{b}"] = float(dist)

    return {"matrices": matrices, "frobenius_distances": frobenius}


def generate_context_report(
    grouped_codes: dict[str, np.ndarray],
    K: int,
    config: "AnalysisConfig",
) -> str:
    """Generate a text report of context-dependent analysis.

    Parameters
    ----------
    grouped_codes:
        Dict mapping group label -> 1D code array.
    K:
        Codebook size.
    config:
        AnalysisConfig.

    Returns
    -------
    Formatted report string.
    """
    lines = ["# Context-Dependent Code Analysis", ""]

    groups = list(grouped_codes.keys())
    lines.append(f"Groups: {len(groups)}")
    for g in groups:
        lines.append(f"  - {g}: {len(grouped_codes[g])} frames")
    lines.append("")

    if len(groups) < 2:
        lines.append("**Note:** Only one group available. Cannot perform "
                      "between-group comparisons.")
        lines.append("")
        lines.append("Richer metadata (sex, strain, social context) would "
                      "enable group comparisons.")
        return "\n".join(lines)

    # Differential usage
    diffs = differential_code_usage(grouped_codes, K)
    lines.append("## Pairwise Comparisons")
    lines.append("")
    for d in diffs:
        lines.append(f"### {d['group_a']} vs {d['group_b']}")
        lines.append(f"  Chi-squared: {d['chi2']:.2f} (p={d['p_value']:.4f})")
        lines.append(f"  KL(A||B): {d['kl_ab']:.4f}, KL(B||A): {d['kl_ba']:.4f}")
        lines.append(f"  Top differential codes: {d['top_differential_codes']}")
        lines.append("")

    # Transition matrix comparison
    trans = compare_transition_matrices(grouped_codes, K)
    lines.append("## Transition Matrix Distances (Frobenius)")
    lines.append("")
    for pair, dist in trans["frobenius_distances"].items():
        lines.append(f"  {pair}: {dist:.4f}")
    lines.append("")

    return "\n".join(lines)
