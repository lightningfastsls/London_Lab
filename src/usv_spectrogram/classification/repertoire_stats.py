"""Syllable repertoire statistical analysis for comparing mouse populations.

Computes per-animal syllable proportions, Shannon entropy (repertoire diversity),
transition matrices (sequential structure), and runs PERMANOVA, chi-squared, and
Jensen-Shannon divergence comparisons between populations (wild vs lab).

PERMANOVA is implemented from scratch using scipy pdist to avoid the skbio
C-dependency chain that fails on Windows.

Usage::

    from usv_spectrogram.classification.repertoire_stats import (
        RepertoireConfig, analyze_repertoire,
    )

    config = RepertoireConfig(
        classified_data_path=Path("classified_detections.csv"),
        output_dir=Path("analysis/repertoire"),
    )
    results = analyze_repertoire(config)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must precede pyplot import.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon, pdist, squareform
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RepertoireConfig:
    """Configuration for a repertoire statistical analysis run.

    Parameters
    ----------
    classified_data_path : Path
        CSV with classified detection data (output from Phase 14.2).
    population_column : str
        Column distinguishing populations (e.g. 'wild' vs 'lab').
    animal_id_column : str
        Column identifying individual animals.
    syllable_column : str
        Column with syllable type labels from DeepSqueak.
    time_column : str
        Column with detection time (seconds) for ordering.
    n_permutations : int
        Number of permutations for permutation tests.
    random_seed : int
        Random seed for reproducibility.
    output_dir : Path
        Directory for all output files.
    """

    classified_data_path: Path
    population_column: str = "population"
    animal_id_column: str = "animal_id"
    syllable_column: str = "label"
    time_column: str = "begin_time_s"
    n_permutations: int = 1000
    random_seed: int = 42
    output_dir: Path = Path("analysis/repertoire")

    def __post_init__(self) -> None:
        # Allow string paths for convenience.
        for attr in ("classified_data_path", "output_dir"):
            val = getattr(self, attr)
            if not isinstance(val, Path):
                object.__setattr__(self, attr, Path(val))

        if self.n_permutations <= 0:
            raise ValueError(
                f"n_permutations must be positive, got {self.n_permutations}"
            )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_required_columns(
    df: pd.DataFrame,
    columns: list[str],
    context: str,
) -> None:
    """Raise ValueError listing any missing columns."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{context}: missing required columns {missing}. "
            f"Available: {list(df.columns)}"
        )


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def syllable_proportions(
    df: pd.DataFrame,
    group_col: str,
    syllable_col: str = "label",
) -> pd.DataFrame:
    """Per-group syllable type proportions.

    Parameters
    ----------
    df : DataFrame
        Classified detection data.
    group_col : str
        Column to group by (typically animal_id).
    syllable_col : str
        Column containing syllable type labels.

    Returns
    -------
    DataFrame
        Columns: [group_col, syllable_type, proportion, count].
        Each group's proportions sum to 1.0.
    """
    _validate_required_columns(df, [group_col, syllable_col], "syllable_proportions")

    records: list[dict] = []
    for group_name, group_df in df.groupby(group_col, sort=True):
        counts = group_df[syllable_col].value_counts()
        total = counts.sum()
        for syl_type, count in counts.items():
            records.append({
                group_col: group_name,
                "syllable_type": syl_type,
                "proportion": count / total,
                "count": int(count),
            })

    return pd.DataFrame(records)


def syllable_diversity(
    df: pd.DataFrame,
    group_col: str,
    syllable_col: str = "label",
) -> pd.DataFrame:
    """Shannon entropy (H) per group — measures repertoire diversity.

    H = -sum(p_i * log2(p_i)) for each group's syllable distribution.
    Higher H = more diverse repertoire. H=0 for single type, H=log2(K)
    for uniform distribution over K types.

    Parameters
    ----------
    df : DataFrame
        Classified detection data.
    group_col : str
        Column to group by (typically animal_id).
    syllable_col : str
        Column containing syllable type labels.

    Returns
    -------
    DataFrame
        Columns: [group_col, entropy_h, n_types_used, n_calls].
    """
    props = syllable_proportions(df, group_col, syllable_col)

    records: list[dict] = []
    for group_name, group_df in props.groupby(group_col, sort=True):
        p = group_df["proportion"].values
        # Shannon entropy: -sum(p * log2(p)) for p > 0
        p_nonzero = p[p > 0]
        h = -np.sum(p_nonzero * np.log2(p_nonzero))

        records.append({
            group_col: group_name,
            "entropy_h": float(h),
            "n_types_used": len(p_nonzero),
            "n_calls": int(group_df["count"].sum()),
        })

    return pd.DataFrame(records)


def transition_matrix(
    df: pd.DataFrame,
    animal_id: str,
    syllable_col: str = "label",
    time_col: str = "begin_time_s",
    animal_id_col: str = "animal_id",
    all_labels: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Syllable-to-syllable transition probability matrix for one animal.

    Sorts detections by time and computes P(type_{t+1} | type_t) from
    consecutive bigrams.

    Parameters
    ----------
    df : DataFrame
        Classified detection data.
    animal_id : str
        Which animal to compute for.
    syllable_col : str
        Column containing syllable type labels.
    time_col : str
        Column for temporal ordering.
    animal_id_col : str
        Column identifying animals.
    all_labels : list[str] or None
        If provided, ensures the matrix uses this exact label set and order
        for consistent dimensions across animals.

    Returns
    -------
    (K x K ndarray, list[str])
        Row-stochastic transition matrix and label list.
        Rows with zero counts stay at 0.0.
    """
    _validate_required_columns(
        df, [animal_id_col, syllable_col, time_col], "transition_matrix"
    )

    animal_df = df[df[animal_id_col] == animal_id].sort_values(time_col)
    labels_seq = animal_df[syllable_col].values

    # Determine label set.
    if all_labels is not None:
        labels = list(all_labels)
    else:
        labels = sorted(set(labels_seq))

    K = len(labels)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    counts = np.zeros((K, K), dtype=np.float64)

    # Build bigram counts from consecutive pairs.
    for i in range(len(labels_seq) - 1):
        curr = labels_seq[i]
        nxt = labels_seq[i + 1]
        if curr in label_to_idx and nxt in label_to_idx:
            counts[label_to_idx[curr], label_to_idx[nxt]] += 1

    # Row-normalize to get probabilities; zero-count rows stay 0.0.
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = np.where(row_sums > 0, counts / row_sums, 0.0)

    return matrix, labels


# ---------------------------------------------------------------------------
# Statistical comparison methods
# ---------------------------------------------------------------------------

def _compute_pseudo_f(
    D_sq: np.ndarray,
    groups: np.ndarray,
    unique_groups: np.ndarray,
    N: int,
) -> float:
    """Compute PERMANOVA pseudo-F statistic from squared distance matrix.

    Parameters
    ----------
    D_sq : (N, N) ndarray
        Squared pairwise distance matrix.
    groups : (N,) ndarray
        Group label for each sample.
    unique_groups : ndarray
        Unique group labels.
    N : int
        Number of samples.

    Returns
    -------
    float
        Pseudo-F statistic. Returns 0.0 if SS_within == 0.
    """
    a = len(unique_groups)

    # SS_total = sum(D^2) / (2*N)
    ss_total = D_sq.sum() / (2 * N)

    # SS_within = sum over groups of: sum(D^2 within group) / (2 * n_g)
    ss_within = 0.0
    for g in unique_groups:
        mask = groups == g
        n_g = mask.sum()
        if n_g < 2:
            continue
        within_dists = D_sq[np.ix_(mask, mask)]
        ss_within += within_dists.sum() / (2 * n_g)

    ss_between = ss_total - ss_within

    # pseudo-F = (SS_between / (a-1)) / (SS_within / (N-a))
    denom = ss_within / (N - a) if (N - a) > 0 else 0.0
    if denom == 0.0:
        return 0.0

    return (ss_between / (a - 1)) / denom


def _permanova_test(
    proportion_vectors: np.ndarray,
    groups: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict:
    """PERMANOVA on Bray-Curtis dissimilarity (Anderson 2001).

    Parameters
    ----------
    proportion_vectors : (N, K) ndarray
        Per-animal syllable proportion vectors.
    groups : (N,) ndarray
        Population label per animal.
    n_permutations : int
        Number of permutation iterations.
    rng : numpy Generator
        Random number generator for reproducibility.

    Returns
    -------
    dict
        {method, statistic, p_value, effect_size, interpretation}
    """
    N = len(groups)
    unique_groups = np.unique(groups)

    # Pairwise Bray-Curtis distances.
    dists = pdist(proportion_vectors, metric="braycurtis")
    D = squareform(dists)
    D_sq = D ** 2

    # Observed pseudo-F.
    f_obs = _compute_pseudo_f(D_sq, groups, unique_groups, N)

    # Effect size: R² = SS_between / SS_total.
    ss_total = D_sq.sum() / (2 * N)
    ss_within = 0.0
    for g in unique_groups:
        mask = groups == g
        n_g = mask.sum()
        if n_g < 2:
            continue
        ss_within += D_sq[np.ix_(mask, mask)].sum() / (2 * n_g)
    r_squared = (ss_total - ss_within) / ss_total if ss_total > 0 else 0.0

    # Permutation test.
    count_ge = 0
    for _ in range(n_permutations):
        perm_groups = rng.permutation(groups)
        f_perm = _compute_pseudo_f(D_sq, perm_groups, unique_groups, N)
        if f_perm >= f_obs:
            count_ge += 1

    p_value = (count_ge + 1) / (n_permutations + 1)

    if p_value < 0.05:
        interp = (
            f"Significant difference (p={p_value:.4f}, R²={r_squared:.3f}): "
            f"syllable repertoire composition differs between populations."
        )
    else:
        interp = (
            f"No significant difference (p={p_value:.4f}, R²={r_squared:.3f}): "
            f"cannot reject that populations have similar repertoires."
        )

    return {
        "method": "permanova",
        "statistic": float(f_obs),
        "p_value": float(p_value),
        "effect_size": float(r_squared),
        "interpretation": interp,
    }


def _jsd_test(
    proportion_vectors: np.ndarray,
    groups: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict:
    """Jensen-Shannon Divergence between pooled population distributions.

    Parameters
    ----------
    proportion_vectors : (N, K) ndarray
        Per-animal syllable proportion vectors.
    groups : (N,) ndarray
        Population label per animal.
    n_permutations : int
        Number of permutation iterations for p-value.
    rng : numpy Generator
        Random number generator.

    Returns
    -------
    dict
        {method, statistic, p_value, effect_size, interpretation}
    """
    unique_groups = np.unique(groups)

    if len(unique_groups) != 2:
        raise ValueError(
            f"JSD requires exactly 2 populations, got {len(unique_groups)}: "
            f"{list(unique_groups)}. Use method='permanova' or 'chi_square' "
            f"for other numbers of populations."
        )

    # Pool proportions per population (mean across animals).
    pop_means = {}
    for g in unique_groups:
        mask = groups == g
        pop_means[g] = proportion_vectors[mask].mean(axis=0)

    g1, g2 = unique_groups[0], unique_groups[1]
    p, q = pop_means[g1], pop_means[g2]

    # JSD = jensenshannon()² (scipy returns distance = sqrt(divergence)).
    jsd_obs = float(jensenshannon(p, q, base=2.0) ** 2)

    # Permutation test.
    count_ge = 0
    for _ in range(n_permutations):
        perm_groups = rng.permutation(groups)
        pm = {}
        for g in unique_groups:
            mask = perm_groups == g
            pm[g] = proportion_vectors[mask].mean(axis=0)
        jsd_perm = float(jensenshannon(pm[g1], pm[g2], base=2.0) ** 2)
        if jsd_perm >= jsd_obs:
            count_ge += 1

    p_value = (count_ge + 1) / (n_permutations + 1)

    if p_value < 0.05:
        interp = (
            f"Significant divergence (JSD={jsd_obs:.4f}, p={p_value:.4f}): "
            f"population syllable distributions differ."
        )
    else:
        interp = (
            f"No significant divergence (JSD={jsd_obs:.4f}, p={p_value:.4f}): "
            f"population distributions are not distinguishable."
        )

    return {
        "method": "jsd",
        "statistic": float(jsd_obs),
        "p_value": float(p_value),
        "effect_size": float(jsd_obs),
        "interpretation": interp,
    }


def _chi_square_test(
    df: pd.DataFrame,
    population_col: str,
    syllable_col: str,
) -> dict:
    """Chi-squared test on pooled syllable counts between populations.

    Parameters
    ----------
    df : DataFrame
        Classified detection data.
    population_col : str
        Column with population labels.
    syllable_col : str
        Column with syllable labels.

    Returns
    -------
    dict
        {method, statistic, p_value, effect_size, interpretation}
    """
    contingency = pd.crosstab(df[population_col], df[syllable_col])
    chi2, p_value, dof, _ = chi2_contingency(contingency)

    # Cramer's V effect size.
    n = contingency.values.sum()
    min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else 0.0

    if p_value < 0.05:
        interp = (
            f"Significant difference (chi2={chi2:.2f}, p={p_value:.4g}, "
            f"Cramer's V={cramers_v:.3f}): syllable usage differs between populations."
        )
    else:
        interp = (
            f"No significant difference (chi2={chi2:.2f}, p={p_value:.4g}, "
            f"Cramer's V={cramers_v:.3f}): syllable usage is similar."
        )

    return {
        "method": "chi_square",
        "statistic": float(chi2),
        "p_value": float(p_value),
        "effect_size": float(cramers_v),
        "interpretation": interp,
    }


def compare_repertoires(
    df: pd.DataFrame,
    population_column: str = "population",
    animal_id_column: str = "animal_id",
    syllable_column: str = "label",
    method: str = "permanova",
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """Compare syllable repertoire distributions between populations.

    Parameters
    ----------
    df : DataFrame
        Classified detection data.
    population_column : str
        Column distinguishing populations.
    animal_id_column : str
        Column identifying individual animals.
    syllable_column : str
        Column with syllable labels.
    method : str
        One of 'permanova', 'chi_square', 'jsd'.
    n_permutations : int
        Permutation count for permutation-based tests.
    seed : int
        Random seed.

    Returns
    -------
    dict
        {method, statistic, p_value, effect_size, interpretation}
    """
    _validate_required_columns(
        df, [population_column, animal_id_column, syllable_column],
        "compare_repertoires",
    )

    if method == "chi_square":
        return _chi_square_test(df, population_column, syllable_column)

    # Build per-animal proportion vectors for permutation-based tests.
    rng = np.random.default_rng(seed)
    props = syllable_proportions(df, animal_id_column, syllable_column)

    # Pivot to animal × syllable_type matrix, filling missing types with 0.
    pivot = props.pivot(
        index=animal_id_column, columns="syllable_type", values="proportion"
    ).fillna(0.0)

    # Get population label per animal.
    animal_pop = (
        df[[animal_id_column, population_column]]
        .drop_duplicates()
        .set_index(animal_id_column)
    )
    groups = animal_pop.loc[pivot.index, population_column].values

    proportion_vectors = pivot.values

    if method == "permanova":
        return _permanova_test(proportion_vectors, groups, n_permutations, rng)
    elif method == "jsd":
        return _jsd_test(proportion_vectors, groups, n_permutations, rng)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'permanova', 'jsd', or 'chi_square'.")


# ---------------------------------------------------------------------------
# Transition matrix comparison
# ---------------------------------------------------------------------------

def compare_transition_matrices(
    df: pd.DataFrame,
    population_column: str = "population",
    animal_id_column: str = "animal_id",
    syllable_column: str = "label",
    time_column: str = "begin_time_s",
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """Compare transition structure between populations via Frobenius norm.

    Computes per-animal transition matrices, averages within each population,
    and tests whether the Frobenius distance between population means is
    larger than expected under shuffled population labels.

    Parameters
    ----------
    df : DataFrame
        Classified detection data.
    population_column, animal_id_column, syllable_column, time_column : str
        Column names.
    n_permutations : int
        Number of permutation iterations.
    seed : int
        Random seed.

    Returns
    -------
    dict
        {frobenius_norm, p_value, differentially_used_transitions, interpretation}
    """
    _validate_required_columns(
        df,
        [population_column, animal_id_column, syllable_column, time_column],
        "compare_transition_matrices",
    )

    rng = np.random.default_rng(seed)

    # Global label set for consistent matrix dimensions.
    all_labels = sorted(df[syllable_column].unique())
    animals = df[animal_id_column].unique()

    # Build per-animal transition matrices.
    animal_matrices: dict[str, np.ndarray] = {}
    animal_pops: dict[str, str] = {}
    for animal in animals:
        mat, _ = transition_matrix(
            df, animal, syllable_column, time_column, animal_id_column, all_labels
        )
        animal_matrices[animal] = mat
        pop = df.loc[df[animal_id_column] == animal, population_column].iloc[0]
        animal_pops[animal] = pop

    unique_pops = sorted(set(animal_pops.values()))

    if len(unique_pops) != 2:
        raise ValueError(
            f"Transition comparison requires exactly 2 populations, "
            f"got {len(unique_pops)}: {unique_pops}. "
            f"Filter your data to two populations before calling."
        )

    animal_list = list(animals)
    pop_labels = np.array([animal_pops[a] for a in animal_list])

    def _mean_matrices(pop_labs: np.ndarray) -> dict[str, np.ndarray]:
        means = {}
        for pop in unique_pops:
            mask = pop_labs == pop
            mats = [animal_matrices[animal_list[i]] for i in range(len(mask)) if mask[i]]
            if mats:
                means[pop] = np.mean(mats, axis=0)
            else:
                means[pop] = np.zeros_like(next(iter(animal_matrices.values())))
        return means

    def _frobenius(pop_labs: np.ndarray) -> float:
        means = _mean_matrices(pop_labs)
        m_list = [means[p] for p in unique_pops]
        return float(np.linalg.norm(m_list[0] - m_list[1], "fro"))

    # Observed Frobenius distance.
    frob_obs = _frobenius(pop_labels)

    # Permutation test.
    count_ge = 0
    for _ in range(n_permutations):
        perm_labels = rng.permutation(pop_labels)
        frob_perm = _frobenius(perm_labels)
        if frob_perm >= frob_obs:
            count_ge += 1

    p_value = (count_ge + 1) / (n_permutations + 1)

    # Top 10 differentially used transitions.
    obs_means = _mean_matrices(pop_labels)
    diff_matrix = obs_means[unique_pops[0]] - obs_means[unique_pops[1]]
    diffs: list[dict] = []
    for i, lab_from in enumerate(all_labels):
        for j, lab_to in enumerate(all_labels):
            diffs.append({
                "from": lab_from,
                "to": lab_to,
                "difference": float(diff_matrix[i, j]),
                "abs_difference": float(abs(diff_matrix[i, j])),
            })
    diffs.sort(key=lambda x: x["abs_difference"], reverse=True)
    top_diffs = [
        {k: v for k, v in d.items() if k != "abs_difference"}
        for d in diffs[:10]
    ]

    if p_value < 0.05:
        interp = (
            f"Significant transition difference (||F||={frob_obs:.4f}, p={p_value:.4f}): "
            f"sequential structure differs between populations."
        )
    else:
        interp = (
            f"No significant transition difference (||F||={frob_obs:.4f}, p={p_value:.4f}): "
            f"sequential structure is similar between populations."
        )

    return {
        "frobenius_norm": float(frob_obs),
        "p_value": float(p_value),
        "differentially_used_transitions": top_diffs,
        "interpretation": interp,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_repertoire_comparison(
    df: pd.DataFrame,
    population_column: str = "population",
    animal_id_column: str = "animal_id",
    syllable_column: str = "label",
    time_column: str = "begin_time_s",
    output_dir: Path = Path("analysis/repertoire"),
) -> list[Path]:
    """Generate publication-ready repertoire comparison figures.

    Creates four plots:
    1. Stacked bar chart of syllable proportions per population
    2. Shannon entropy box + strip plot per population
    3. Side-by-side transition heatmaps per population
    4. PCA scatter of animal-level proportion vectors

    Parameters
    ----------
    df : DataFrame
        Classified detection data.
    population_column, animal_id_column, syllable_column, time_column : str
        Column names.
    output_dir : Path
        Directory for saving figures.

    Returns
    -------
    list[Path]
        Paths to created figure files.
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    populations = sorted(df[population_column].unique())

    # --- 1. Stacked bar chart: syllable proportions per population ---
    pop_props = syllable_proportions(df, population_column, syllable_column)
    pivot = pop_props.pivot(
        index=population_column, columns="syllable_type", values="proportion"
    ).fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Population")
    ax.set_title("Syllable Type Proportions by Population")
    ax.legend(title="Syllable Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylim(0, 1)
    path = figures_dir / "syllable_proportions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    # --- 2. Diversity box + strip plot ---
    diversity = syllable_diversity(df, animal_id_column, syllable_column)
    # Merge population labels.
    animal_pop = (
        df[[animal_id_column, population_column]]
        .drop_duplicates()
        .set_index(animal_id_column)
    )
    diversity = diversity.merge(
        animal_pop, left_on=animal_id_column, right_index=True, how="left"
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    pop_data = [
        diversity.loc[diversity[population_column] == pop, "entropy_h"].values
        for pop in populations
    ]
    bp = ax.boxplot(pop_data, tick_labels=populations, patch_artist=True)
    colors = ["#4C72B0", "#DD8452"]
    for patch, color in zip(bp["boxes"], colors[:len(populations)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    # Strip plot overlay.
    for i, (pop, data) in enumerate(zip(populations, pop_data)):
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, size=len(data))
        ax.scatter(
            np.full(len(data), i + 1) + jitter, data,
            alpha=0.7, s=30, zorder=3, color=colors[i % len(colors)],
        )

    # Mann-Whitney U test annotation.
    if len(populations) == 2 and len(pop_data[0]) > 0 and len(pop_data[1]) > 0:
        stat, mwu_p = mannwhitneyu(pop_data[0], pop_data[1], alternative="two-sided")
        ax.set_title(f"Repertoire Diversity (Mann-Whitney U p={mwu_p:.4f})")
    else:
        ax.set_title("Repertoire Diversity (Shannon Entropy)")
    ax.set_ylabel("Shannon Entropy H (bits)")
    ax.set_xlabel("Population")
    path = figures_dir / "diversity_boxplot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    # --- 3. Transition heatmaps side by side ---
    all_labels = sorted(df[syllable_column].unique())
    K = len(all_labels)

    fig, axes = plt.subplots(1, len(populations), figsize=(5 * len(populations) + 2, 5))
    if len(populations) == 1:
        axes = [axes]

    vmin, vmax = 0.0, 1.0
    for ax_i, pop in zip(axes, populations):
        pop_animals = df.loc[df[population_column] == pop, animal_id_column].unique()
        if len(pop_animals) == 0:
            continue
        mats = []
        for animal in pop_animals:
            mat, _ = transition_matrix(
                df, animal, syllable_column, time_column, animal_id_column, all_labels
            )
            mats.append(mat)
        mean_mat = np.mean(mats, axis=0)

        im = ax_i.imshow(mean_mat, cmap="YlOrRd", vmin=vmin, vmax=vmax, aspect="auto")
        ax_i.set_xticks(range(K))
        ax_i.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=7)
        ax_i.set_yticks(range(K))
        ax_i.set_yticklabels(all_labels, fontsize=7)
        ax_i.set_title(f"{pop} (n={len(pop_animals)})")
        ax_i.set_xlabel("Next syllable")
        ax_i.set_ylabel("Current syllable")

    fig.colorbar(im, ax=axes, shrink=0.8, label="Transition probability")
    fig.suptitle("Mean Transition Matrices by Population", fontsize=12)
    path = figures_dir / "transition_heatmaps.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    # --- 4. PCA scatter of animal proportions ---
    animal_props = syllable_proportions(df, animal_id_column, syllable_column)
    pivot_animal = animal_props.pivot(
        index=animal_id_column, columns="syllable_type", values="proportion"
    ).fillna(0.0)
    animal_pop_map = (
        df[[animal_id_column, population_column]]
        .drop_duplicates()
        .set_index(animal_id_column)
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    if pivot_animal.shape[0] >= 2 and pivot_animal.shape[1] >= 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(pivot_animal.values)
        pop_labels_arr = animal_pop_map.loc[pivot_animal.index, population_column].values

        for i, pop in enumerate(populations):
            mask = pop_labels_arr == pop
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                label=pop, alpha=0.7, s=50, color=colors[i % len(colors)],
            )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    else:
        # Too few animals or syllable types for PCA.
        ax.text(0.5, 0.5, "Too few animals or syllable types for PCA",
                ha="center", va="center", transform=ax.transAxes)

    ax.set_title("Animal-Level Syllable Proportions (PCA)")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(title="Population")
    path = figures_dir / "animal_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    return created


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    results: dict,
    output_dir: Path,
) -> Path:
    """Write a plain-language repertoire analysis report in markdown.

    Parameters
    ----------
    results : dict
        The full results dict from :func:`analyze_repertoire`.
    output_dir : Path
        Directory to write ``repertoire_report.md``.

    Returns
    -------
    Path
        Path to the written report file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "repertoire_report.md"

    lines = [
        "# Syllable Repertoire Analysis Report",
        "",
        "## Summary",
        "",
    ]

    # Diversity summary.
    if "diversity" in results:
        div_df = results["diversity"]
        lines.append("### Repertoire Diversity (Shannon Entropy)")
        lines.append("")
        lines.append("| Animal | Entropy H | Types Used | Calls |")
        lines.append("|--------|-----------|------------|-------|")
        for _, row in div_df.iterrows():
            lines.append(
                f"| {row.iloc[0]} | {row['entropy_h']:.3f} | "
                f"{row['n_types_used']} | {row['n_calls']} |"
            )
        lines.append("")

    # Statistical tests.
    for test_name in ("permanova", "jsd", "chi_square"):
        if test_name in results:
            res = results[test_name]
            lines.append(f"### {res['method'].upper()} Test")
            lines.append("")
            lines.append(f"- **Statistic:** {res['statistic']:.4f}")
            lines.append(f"- **p-value:** {res['p_value']:.4f}")
            lines.append(f"- **Effect size:** {res['effect_size']:.4f}")
            lines.append(f"- **Interpretation:** {res['interpretation']}")
            lines.append("")

    # Transition comparison.
    if "transition_comparison" in results:
        tc = results["transition_comparison"]
        lines.append("### Transition Matrix Comparison")
        lines.append("")
        lines.append(f"- **Frobenius norm:** {tc['frobenius_norm']:.4f}")
        lines.append(f"- **p-value:** {tc['p_value']:.4f}")
        lines.append(f"- **Interpretation:** {tc['interpretation']}")
        lines.append("")
        if tc.get("differentially_used_transitions"):
            lines.append("**Top differentially used transitions:**")
            lines.append("")
            lines.append("| From | To | Difference |")
            lines.append("|------|----|------------|")
            for t in tc["differentially_used_transitions"][:10]:
                lines.append(f"| {t['from']} | {t['to']} | {t['difference']:+.4f} |")
            lines.append("")

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Report written to %s", report_path)
    return report_path


# ---------------------------------------------------------------------------
# Full pipeline orchestrator
# ---------------------------------------------------------------------------

def analyze_repertoire(config: RepertoireConfig) -> dict:
    """Run the full repertoire statistical analysis pipeline.

    1. Load classified CSV.
    2. Compute proportions and diversity.
    3. Run all three statistical tests.
    4. Compare transition matrices.
    5. Generate plots and report.
    6. Save all outputs.

    Parameters
    ----------
    config : RepertoireConfig
        Validated analysis configuration.

    Returns
    -------
    dict
        Full results including DataFrames, test results, and file paths.
    """
    # -- Load data --
    logger.info("Loading classified data from %s", config.classified_data_path)
    df = pd.read_csv(config.classified_data_path)

    required_cols = [
        config.population_column,
        config.animal_id_column,
        config.syllable_column,
        config.time_column,
    ]
    _validate_required_columns(df, required_cols, "analyze_repertoire")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {}

    # -- Proportions --
    logger.info("Computing syllable proportions...")
    props = syllable_proportions(df, config.animal_id_column, config.syllable_column)
    props_path = config.output_dir / "syllable_proportions.csv"
    props.to_csv(props_path, index=False)
    results["proportions"] = props

    # -- Diversity --
    logger.info("Computing syllable diversity...")
    diversity = syllable_diversity(df, config.animal_id_column, config.syllable_column)
    div_path = config.output_dir / "diversity_comparison.csv"
    diversity.to_csv(div_path, index=False)
    results["diversity"] = diversity

    # -- Statistical tests --
    n_pops = df[config.population_column].nunique()

    # PERMANOVA and chi-square work with any number of populations.
    # JSD requires exactly 2.
    for method in ("permanova", "chi_square"):
        logger.info("Running %s test...", method)
        results[method] = compare_repertoires(
            df,
            population_column=config.population_column,
            animal_id_column=config.animal_id_column,
            syllable_column=config.syllable_column,
            method=method,
            n_permutations=config.n_permutations,
            seed=config.random_seed,
        )

    if n_pops == 2:
        logger.info("Running jsd test...")
        results["jsd"] = compare_repertoires(
            df,
            population_column=config.population_column,
            animal_id_column=config.animal_id_column,
            syllable_column=config.syllable_column,
            method="jsd",
            n_permutations=config.n_permutations,
            seed=config.random_seed,
        )
    else:
        logger.warning(
            "JSD requires exactly 2 populations (found %d). Skipping.", n_pops
        )

    # -- Transition comparison (requires exactly 2 populations) --
    if n_pops == 2:
        logger.info("Comparing transition matrices...")
        results["transition_comparison"] = compare_transition_matrices(
            df,
            population_column=config.population_column,
            animal_id_column=config.animal_id_column,
            syllable_column=config.syllable_column,
            time_column=config.time_column,
            n_permutations=config.n_permutations,
            seed=config.random_seed,
        )
    else:
        logger.warning(
            "Transition matrix comparison requires exactly 2 populations "
            "(found %d). Skipping.", n_pops
        )

    # Save transition matrices per population.
    trans_dir = config.output_dir / "transition_matrices"
    trans_dir.mkdir(parents=True, exist_ok=True)
    all_labels = sorted(df[config.syllable_column].unique())
    for pop in df[config.population_column].unique():
        pop_animals = df.loc[
            df[config.population_column] == pop, config.animal_id_column
        ].unique()
        mats = []
        for animal in pop_animals:
            mat, _ = transition_matrix(
                df, animal, config.syllable_column,
                config.time_column, config.animal_id_column, all_labels,
            )
            mats.append(mat)
        mean_mat = np.mean(mats, axis=0)
        mat_df = pd.DataFrame(mean_mat, index=all_labels, columns=all_labels)
        mat_df.to_csv(trans_dir / f"{pop}_transition_matrix.csv")

    # -- Plots --
    logger.info("Generating plots...")
    figure_paths = plot_repertoire_comparison(
        df,
        population_column=config.population_column,
        animal_id_column=config.animal_id_column,
        syllable_column=config.syllable_column,
        time_column=config.time_column,
        output_dir=config.output_dir,
    )
    results["figure_paths"] = figure_paths

    # -- Save all test results as JSON --
    test_results = {}
    for key in ("permanova", "jsd", "chi_square", "transition_comparison"):
        if key in results:
            test_results[key] = results[key]
    tests_path = config.output_dir / "statistical_tests.json"
    with open(tests_path, "w", encoding="utf-8") as fh:
        json.dump(test_results, fh, indent=2, default=str)
    results["tests_path"] = tests_path

    # -- Report --
    logger.info("Writing report...")
    report_path = generate_report(results, config.output_dir)
    results["report_path"] = report_path

    logger.info("Analysis complete. Results in %s", config.output_dir)
    return results
