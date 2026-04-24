"""Cross-population comparison for USV classified-detection CSVs.

Pairwise statistical comparison between two populations of classified USV
calls (e.g. wild vs lab, or two wild cohorts). Delegates to the canonical
primitives already in the project and adds the glue plus a handful of new
metrics (Cohen's h per type, feature KS + Cohen's d, joint-UMAP overlap).

Minimal usage::

    from usv_spectrogram.classification.cross_population import (
        CrossPopulationComparison,
    )

    cmp = CrossPopulationComparison(
        pop_a_csv="results/traditional_taxonomy/classified_traditional.csv",
        pop_a_label="wild_5970",
        pop_b_csv="results/traditional_taxonomy_3452/classified_traditional.csv",
        pop_b_label="wild_3452",
        bout_threshold_s=0.6,
    )
    report = cmp.run_all()
    report.write_json("results/cross_population/wild_5970_vs_wild_3452.json")
    report.write_markdown("results/cross_population/wild_5970_vs_wild_3452.md")
    report.write_figures("results/cross_population/wild_5970_vs_wild_3452/")

Canonical primitives reused:
    - repertoire_stats.syllable_diversity    Shannon entropy
    - sequence_analysis.segment_into_bouts   bout segmentation (strict '>')
    - sequence_analysis.mutual_information_within_bouts   bout-aware MI
    - information_theory.zipf_exponent_mle   Zipf MLE (Clauset 2009)
    - scipy.spatial.distance.jensenshannon   JSD (distance, squared here)
    - scipy.stats.chi2_contingency, ks_2samp standard tests
    - scipy.stats.gaussian_kde                joint-UMAP density

Bout threshold: 0.6 s, matches ``scattoni_7_bout_aware`` in
``data/corpus_facts/5970.json``. Threshold applies to silent-gap ICI
(end-to-start), not IOI.

Sample rate comes from ``usv_spectrogram.corpus.SAMPLE_RATE_HZ``; this module
never redeclares STFT/sample-rate constants.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, gaussian_kde, ks_2samp

from usv_language.analysis.information_theory import ZipfResult, zipf_exponent_mle
from usv_language.analysis.sequence_analysis import (
    mutual_information_within_bouts,
    segment_into_bouts,
)
from usv_spectrogram.corpus import SAMPLE_RATE_HZ  # noqa: F401 (canary: imported not redeclared)

logger = logging.getLogger(__name__)


SCHEMA_VERSION = "1.0"
CANONICAL_BOUT_THRESHOLD_S = 0.6
DEFAULT_FEATURE_COLUMNS: tuple[str, ...] = (
    "principal_freq_hz",
    "low_freq_hz",
    "high_freq_hz",
    "bandwidth_hz",
    "slope",
    "sinuosity",
    "mean_power_db",
    "tonality",
    "call_length_s",
)


# ---------------------------------------------------------------------------
# Typed result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypeProportionResult:
    chi2_statistic: float
    chi2_p_value: float
    chi2_dof: int
    cramers_v: float
    per_type_cohens_h: dict[str, float]
    max_abs_cohens_h: float
    max_abs_cohens_h_type: Optional[str]
    pop_a_counts: dict[str, int]
    pop_b_counts: dict[str, int]
    interpretation: str


@dataclass(frozen=True)
class JSDResult:
    jsd_bits: float
    bootstrap_ci_95: tuple[float, float]
    n_bootstrap: int
    interpretation: str


@dataclass(frozen=True)
class EntropyResult:
    pop_a_entropy_bits: float
    pop_b_entropy_bits: float
    difference_bits: float
    pop_a_ci_95: tuple[float, float]
    pop_b_ci_95: tuple[float, float]
    permutation_p_value: float
    n_bootstrap: int
    n_permutations: int
    interpretation: str


@dataclass(frozen=True)
class TransitionResult:
    labels: list[str]
    pop_a_matrix: list[list[float]]
    pop_b_matrix: list[list[float]]
    pop_a_n_within_pairs: int
    pop_b_n_within_pairs: int
    frobenius_distance: float
    per_row_jsd_bits: dict[str, float]
    interpretation: str


@dataclass(frozen=True)
class MILag1Result:
    pop_a_mi_bits: float
    pop_b_mi_bits: float
    pop_a_n_within_pairs: int
    pop_b_n_within_pairs: int
    pop_a_n_excluded_pairs: int
    pop_b_n_excluded_pairs: int
    pop_a_ci_95: tuple[float, float]
    pop_b_ci_95: tuple[float, float]
    difference_bits: float
    n_bootstrap: int
    interpretation: str


@dataclass(frozen=True)
class ZipfPopResult:
    alpha: float
    rank_alpha: float
    xmin: float
    p_value: float
    n_tail: int
    log_likelihood_ratio: float
    insufficient_types: bool


@dataclass(frozen=True)
class ZipfComparisonResult:
    pop_a: ZipfPopResult
    pop_b: ZipfPopResult
    alpha_difference: float
    interpretation: str


@dataclass(frozen=True)
class BurstinessResult:
    pop_a_cv: float
    pop_b_cv: float
    pop_a_mean_ioi_s: float
    pop_b_mean_ioi_s: float
    pop_a_n_iois: int
    pop_b_n_iois: int
    interpretation: str


@dataclass(frozen=True)
class IOIDistributionResult:
    pop_a_median_s: float
    pop_b_median_s: float
    pop_a_iqr_s: tuple[float, float]
    pop_b_iqr_s: tuple[float, float]
    ks_statistic: float
    ks_p_value: float
    n_a: int
    n_b: int
    interpretation: str


@dataclass(frozen=True)
class FeatureComparison:
    feature: str
    pop_a_mean: float
    pop_b_mean: float
    pop_a_std: float
    pop_b_std: float
    pop_a_median: float
    pop_b_median: float
    cohens_d: float
    ks_statistic: float
    ks_p_value: float
    n_a: int
    n_b: int


@dataclass(frozen=True)
class FeatureComparisonResult:
    per_feature: dict[str, FeatureComparison]
    max_abs_cohens_d_feature: Optional[str]
    max_abs_cohens_d: float


@dataclass(frozen=True)
class UMAPOverlapResult:
    overlap_coefficient: float
    grid_resolution: int
    n_pop_a: int
    n_pop_b: int
    used_subsample: bool
    subsample_n: Optional[int]
    interpretation: str


@dataclass(frozen=True)
class ComparisonMetadata:
    schema_version: str
    pop_a_label: str
    pop_b_label: str
    pop_a_n_calls: int
    pop_b_n_calls: int
    pop_a_n_files: int
    pop_b_n_files: int
    bout_threshold_s: float
    type_column: str
    confidence_column: str
    random_state: int
    n_bootstrap: int
    caveats: list[str]


@dataclass
class ComparisonReport:
    metadata: ComparisonMetadata
    type_proportions: Optional[TypeProportionResult] = None
    jsd: Optional[JSDResult] = None
    entropy: Optional[EntropyResult] = None
    transitions: Optional[TransitionResult] = None
    mi_lag1: Optional[MILag1Result] = None
    zipf: Optional[ZipfComparisonResult] = None
    burstiness: Optional[BurstinessResult] = None
    ioi: Optional[IOIDistributionResult] = None
    features: Optional[FeatureComparisonResult] = None
    umap_overlap: Optional[UMAPOverlapResult] = None
    figure_paths: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """One-screen summary suitable for print()."""
        m = self.metadata
        lines = [
            f"CrossPopulationComparison report (schema {m.schema_version})",
            f"  A = {m.pop_a_label}  N_calls={m.pop_a_n_calls}  N_files={m.pop_a_n_files}",
            f"  B = {m.pop_b_label}  N_calls={m.pop_b_n_calls}  N_files={m.pop_b_n_files}",
            f"  bout_threshold_s = {m.bout_threshold_s}",
        ]
        if self.type_proportions is not None:
            r = self.type_proportions
            lines.append(
                f"  type proportions: chi2={r.chi2_statistic:.2f} "
                f"p={r.chi2_p_value:.4g} V={r.cramers_v:.3f} "
                f"max|h|={r.max_abs_cohens_h:.3f} ({r.max_abs_cohens_h_type})"
            )
        if self.jsd is not None:
            r = self.jsd
            lo, hi = r.bootstrap_ci_95
            lines.append(
                f"  JSD = {r.jsd_bits:.4f} bits [95% CI {lo:.4f}, {hi:.4f}]"
            )
        if self.entropy is not None:
            r = self.entropy
            lines.append(
                f"  Shannon H: A={r.pop_a_entropy_bits:.3f} B={r.pop_b_entropy_bits:.3f} "
                f"diff={r.difference_bits:+.3f} perm_p={r.permutation_p_value:.4g}"
            )
        if self.mi_lag1 is not None:
            r = self.mi_lag1
            lines.append(
                f"  MI lag-1 (bout-aware): A={r.pop_a_mi_bits:.4f} "
                f"B={r.pop_b_mi_bits:.4f} diff={r.difference_bits:+.4f}"
            )
        if self.transitions is not None:
            r = self.transitions
            lines.append(
                f"  Transitions: Frobenius={r.frobenius_distance:.4f} "
                f"(A n_within={r.pop_a_n_within_pairs}, B n_within={r.pop_b_n_within_pairs})"
            )
        if self.zipf is not None:
            r = self.zipf
            lines.append(
                f"  Zipf: A_alpha={r.pop_a.alpha:.3f} B_alpha={r.pop_b.alpha:.3f} "
                f"diff={r.alpha_difference:+.3f}"
            )
        if self.burstiness is not None:
            r = self.burstiness
            lines.append(
                f"  Burstiness CV: A={r.pop_a_cv:.3f} B={r.pop_b_cv:.3f}"
            )
        if self.ioi is not None:
            r = self.ioi
            lines.append(
                f"  IOI median: A={r.pop_a_median_s*1000:.1f}ms "
                f"B={r.pop_b_median_s*1000:.1f}ms KS_p={r.ks_p_value:.4g}"
            )
        if self.features is not None:
            r = self.features
            lines.append(
                f"  Features: max|d|={r.max_abs_cohens_d:.3f} "
                f"({r.max_abs_cohens_d_feature})"
            )
        if self.umap_overlap is not None:
            r = self.umap_overlap
            lines.append(
                f"  Joint-UMAP overlap: {r.overlap_coefficient:.3f} "
                f"(grid={r.grid_resolution})"
            )
        if m.caveats:
            lines.append("  CAVEATS:")
            for c in m.caveats:
                lines.append(f"    - {c}")
        return "\n".join(lines)

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = _report_to_json_dict(self)
        path.write_text(json.dumps(blob, indent=2, default=_json_default))
        return path

    def write_markdown(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_render_markdown(self))
        return path

    def write_figures(self, out_dir: str | Path) -> list[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return _render_figures(self, out_dir)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class CrossPopulationComparison:
    """Pairwise comparison between two classified-detection CSVs.

    See module docstring for usage. All ten ``compare_*`` methods are
    independently callable; ``run_all()`` runs them sequentially and packages
    the results into a :class:`ComparisonReport`.

    Parameters
    ----------
    pop_a_csv, pop_b_csv
        Paths to classified-detection CSVs (columns at minimum:
        ``file, syllable_type, begin_time_s, end_time_s,
        classification_confidence``).
    pop_a_label, pop_b_label
        Short identifiers used in figures and output JSON.
    bout_threshold_s
        Silent-gap threshold for within-bout pair filtering (default 0.6,
        matches canonical ``scattoni_7_bout_aware`` in
        ``data/corpus_facts/5970.json``).
    type_column
        Column with syllable-type labels (default ``"syllable_type"``).
    confidence_column
        Column with classification confidence in [0, 1] (default
        ``"classification_confidence"``).
    file_column, time_column, end_time_column
        Audio-file and event timing columns (defaults: ``"file"``,
        ``"begin_time_s"``, ``"end_time_s"``).
    feature_columns
        Iterable of acoustic-feature column names for ``compare_features``.
        Any column missing from either CSV is skipped with a caveat.
    n_bootstrap
        Number of bootstrap resamples for confidence intervals (default
        1000, percentile method).
    n_permutations
        Permutations for entropy-difference permutation test.
    random_state
        Seed for all stochastic operations.
    confidence_min
        Optional lower-bound filter on ``confidence_column``. ``None``
        keeps all calls.
    """

    def __init__(
        self,
        pop_a_csv: str | Path,
        pop_a_label: str,
        pop_b_csv: str | Path,
        pop_b_label: str,
        bout_threshold_s: float = CANONICAL_BOUT_THRESHOLD_S,
        type_column: str = "syllable_type",
        confidence_column: str = "classification_confidence",
        file_column: str = "file",
        time_column: str = "begin_time_s",
        end_time_column: str = "end_time_s",
        feature_columns: Iterable[str] = DEFAULT_FEATURE_COLUMNS,
        n_bootstrap: int = 1000,
        n_permutations: int = 1000,
        random_state: int = 42,
        confidence_min: Optional[float] = None,
    ) -> None:
        if pop_a_label == pop_b_label:
            raise ValueError(
                f"pop_a_label and pop_b_label must differ, got {pop_a_label!r}"
            )
        if bout_threshold_s <= 0:
            raise ValueError(
                f"bout_threshold_s must be > 0, got {bout_threshold_s!r}"
            )

        self.pop_a_label = pop_a_label
        self.pop_b_label = pop_b_label
        self.bout_threshold_s = float(bout_threshold_s)
        self.type_column = type_column
        self.confidence_column = confidence_column
        self.file_column = file_column
        self.time_column = time_column
        self.end_time_column = end_time_column
        self.feature_columns = tuple(feature_columns)
        self.n_bootstrap = int(n_bootstrap)
        self.n_permutations = int(n_permutations)
        self.random_state = int(random_state)
        self.confidence_min = confidence_min

        self._caveats: list[str] = []
        self._rng = np.random.default_rng(self.random_state)

        self.pop_a_df = self._load(pop_a_csv, pop_a_label)
        self.pop_b_df = self._load(pop_b_csv, pop_b_label)

        # Unified label alphabet across both populations for MI/transition
        # work. Sorted for determinism.
        self.all_labels: list[str] = sorted(
            set(self.pop_a_df[self.type_column].unique())
            | set(self.pop_b_df[self.type_column].unique())
        )

    # -- Loading ------------------------------------------------------------

    def _load(self, csv_path: str | Path, label: str) -> pd.DataFrame:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found for {label}: {path}")
        df = pd.read_csv(path)
        required = [
            self.type_column,
            self.file_column,
            self.time_column,
            self.end_time_column,
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"{label}: missing required columns {missing}. "
                f"Available: {list(df.columns)}"
            )

        n_before = len(df)
        df = df.dropna(subset=[self.file_column, self.type_column]).copy()
        n_dropped = n_before - len(df)
        if n_dropped:
            self._caveats.append(
                f"{label}: dropped {n_dropped} rows with NaN file/type"
            )

        if (
            self.confidence_min is not None
            and self.confidence_column in df.columns
        ):
            n_before = len(df)
            df = df[df[self.confidence_column] >= self.confidence_min].copy()
            self._caveats.append(
                f"{label}: filtered {n_before - len(df)} rows with "
                f"{self.confidence_column} < {self.confidence_min}"
            )

        df = df.sort_values([self.file_column, self.time_column]).reset_index(drop=True)
        return df

    # -- Shared derivations -------------------------------------------------

    def _bout_pairs_per_file(self, df: pd.DataFrame) -> list[np.ndarray]:
        """Return one array of integer codes per bout, across all files.

        Codes are indices into ``self.all_labels``. Cross-file pairs are
        not considered (each file segments independently).
        """
        bouts: list[np.ndarray] = []
        label_to_idx = {lab: i for i, lab in enumerate(self.all_labels)}
        for _, file_df in df.groupby(self.file_column, sort=False):
            if len(file_df) == 0:
                continue
            codes = file_df[self.type_column].map(label_to_idx).to_numpy()
            if np.any(pd.isna(codes)):
                # Defensive; groupby label_to_idx should always hit because
                # all_labels was derived from the union of both pops.
                continue
            codes = codes.astype(np.int64)
            if len(codes) < 2:
                bouts.append(codes)
                continue
            begin = file_df[self.time_column].to_numpy(dtype=np.float64)
            end = file_df[self.end_time_column].to_numpy(dtype=np.float64)
            # ici_gap = next.begin - this.end, all within the same file.
            ici_gap_s = begin[1:] - end[:-1]
            # Guard against negative gaps (overlapping calls) — treat as 0.
            ici_gap_s = np.where(ici_gap_s < 0, 0.0, ici_gap_s)
            bouts.extend(segment_into_bouts(codes, ici_gap_s, self.bout_threshold_s))
        return bouts

    def _iois_within_bouts(self, df: pd.DataFrame) -> np.ndarray:
        """Within-bout onset-to-onset intervals in seconds, all files pooled."""
        all_iois: list[np.ndarray] = []
        for _, file_df in df.groupby(self.file_column, sort=False):
            if len(file_df) < 2:
                continue
            begin = file_df[self.time_column].to_numpy(dtype=np.float64)
            end = file_df[self.end_time_column].to_numpy(dtype=np.float64)
            ici_gap = begin[1:] - end[:-1]
            ioi = np.diff(begin)
            # within-bout mask: ici_gap <= threshold (strict '>' defines bout
            # boundary, mirroring segment_into_bouts).
            mask = ici_gap <= self.bout_threshold_s
            mask &= ioi >= 0
            all_iois.append(ioi[mask])
        if not all_iois:
            return np.array([], dtype=np.float64)
        return np.concatenate(all_iois)

    def _proportions(self, df: pd.DataFrame) -> pd.Series:
        counts = df[self.type_column].value_counts()
        aligned = counts.reindex(self.all_labels, fill_value=0).astype(float)
        total = aligned.sum()
        if total == 0:
            return aligned
        return aligned / total

    # -- Individual metrics -------------------------------------------------

    def compare_type_proportions(self) -> TypeProportionResult:
        counts_a = (
            self.pop_a_df[self.type_column]
            .value_counts()
            .reindex(self.all_labels, fill_value=0)
        )
        counts_b = (
            self.pop_b_df[self.type_column]
            .value_counts()
            .reindex(self.all_labels, fill_value=0)
        )
        contingency = np.vstack([counts_a.values, counts_b.values])
        chi2, p, dof, _ = chi2_contingency(contingency)

        n_total = contingency.sum()
        min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
        cramers_v = (
            float(np.sqrt(chi2 / (n_total * min_dim))) if min_dim > 0 else 0.0
        )

        p_a = self._proportions(self.pop_a_df)
        p_b = self._proportions(self.pop_b_df)
        # Cohen's h = 2 * (arcsin(sqrt(p_a)) - arcsin(sqrt(p_b)))
        with np.errstate(invalid="ignore"):
            h = 2.0 * (np.arcsin(np.sqrt(p_a.values)) - np.arcsin(np.sqrt(p_b.values)))
        per_type_h = {
            lab: float(h_i) for lab, h_i in zip(self.all_labels, h)
        }
        max_type: Optional[str] = None
        max_abs = 0.0
        for lab, h_i in per_type_h.items():
            if abs(h_i) > max_abs:
                max_abs = abs(h_i)
                max_type = lab

        interp = (
            f"chi2={chi2:.2f} p={p:.4g} Cramer's V={cramers_v:.3f}; "
            f"largest per-type Cohen's h = {max_abs:.3f} on '{max_type}'"
        )

        return TypeProportionResult(
            chi2_statistic=float(chi2),
            chi2_p_value=float(p),
            chi2_dof=int(dof),
            cramers_v=cramers_v,
            per_type_cohens_h=per_type_h,
            max_abs_cohens_h=float(max_abs),
            max_abs_cohens_h_type=max_type,
            pop_a_counts={lab: int(counts_a[lab]) for lab in self.all_labels},
            pop_b_counts={lab: int(counts_b[lab]) for lab in self.all_labels},
            interpretation=interp,
        )

    def compare_repertoires_jsd(self) -> JSDResult:
        p_a = self._proportions(self.pop_a_df).values
        p_b = self._proportions(self.pop_b_df).values
        jsd_obs = float(jensenshannon(p_a, p_b, base=2.0) ** 2)

        # Bootstrap CI: resample calls per population, recompute JSD.
        n_a = len(self.pop_a_df)
        n_b = len(self.pop_b_df)
        labels_a = self.pop_a_df[self.type_column].values
        labels_b = self.pop_b_df[self.type_column].values
        label_to_idx = {lab: i for i, lab in enumerate(self.all_labels)}
        codes_a = np.array([label_to_idx[l] for l in labels_a])
        codes_b = np.array([label_to_idx[l] for l in labels_b])
        K = len(self.all_labels)

        samples = np.empty(self.n_bootstrap, dtype=np.float64)
        for i in range(self.n_bootstrap):
            idx_a = self._rng.integers(0, n_a, size=n_a)
            idx_b = self._rng.integers(0, n_b, size=n_b)
            boot_a = np.bincount(codes_a[idx_a], minlength=K).astype(float)
            boot_b = np.bincount(codes_b[idx_b], minlength=K).astype(float)
            boot_a /= max(boot_a.sum(), 1.0)
            boot_b /= max(boot_b.sum(), 1.0)
            samples[i] = float(jensenshannon(boot_a, boot_b, base=2.0) ** 2)
        ci = (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))
        interp = (
            f"JSD = {jsd_obs:.4f} bits [95% CI {ci[0]:.4f}, {ci[1]:.4f}] "
            f"(higher = more divergent; 0 = identical distributions)"
        )
        return JSDResult(
            jsd_bits=jsd_obs,
            bootstrap_ci_95=ci,
            n_bootstrap=self.n_bootstrap,
            interpretation=interp,
        )

    def compare_entropy(self) -> EntropyResult:
        def shannon(p: np.ndarray) -> float:
            p_nz = p[p > 0]
            return float(-np.sum(p_nz * np.log2(p_nz)))

        p_a = self._proportions(self.pop_a_df).values
        p_b = self._proportions(self.pop_b_df).values
        h_a = shannon(p_a)
        h_b = shannon(p_b)
        diff_obs = h_a - h_b

        K = len(self.all_labels)
        label_to_idx = {lab: i for i, lab in enumerate(self.all_labels)}
        codes_a = np.array(
            [label_to_idx[l] for l in self.pop_a_df[self.type_column].values]
        )
        codes_b = np.array(
            [label_to_idx[l] for l in self.pop_b_df[self.type_column].values]
        )
        n_a = len(codes_a)
        n_b = len(codes_b)

        def bootstrap_h(codes: np.ndarray, n: int) -> tuple[float, float]:
            samples = np.empty(self.n_bootstrap)
            for i in range(self.n_bootstrap):
                idx = self._rng.integers(0, n, size=n)
                boot = np.bincount(codes[idx], minlength=K).astype(float)
                boot /= max(boot.sum(), 1.0)
                samples[i] = shannon(boot)
            return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))

        ci_a = bootstrap_h(codes_a, n_a)
        ci_b = bootstrap_h(codes_b, n_b)

        combined = np.concatenate([codes_a, codes_b])
        perm_count = 0
        for _ in range(self.n_permutations):
            self._rng.shuffle(combined)
            pa = np.bincount(combined[:n_a], minlength=K).astype(float)
            pb = np.bincount(combined[n_a:], minlength=K).astype(float)
            pa /= max(pa.sum(), 1.0)
            pb /= max(pb.sum(), 1.0)
            diff_perm = shannon(pa) - shannon(pb)
            if abs(diff_perm) >= abs(diff_obs):
                perm_count += 1
        perm_p = (perm_count + 1) / (self.n_permutations + 1)

        interp = (
            f"H(A)={h_a:.3f} bits, H(B)={h_b:.3f} bits, diff={diff_obs:+.3f}, "
            f"permutation p={perm_p:.4g} "
            f"(max uniform entropy for K={K} types = {np.log2(K):.3f} bits)"
        )
        return EntropyResult(
            pop_a_entropy_bits=h_a,
            pop_b_entropy_bits=h_b,
            difference_bits=float(diff_obs),
            pop_a_ci_95=ci_a,
            pop_b_ci_95=ci_b,
            permutation_p_value=float(perm_p),
            n_bootstrap=self.n_bootstrap,
            n_permutations=self.n_permutations,
            interpretation=interp,
        )

    def compare_transitions(self) -> TransitionResult:
        K = len(self.all_labels)

        def matrix_from_bouts(bouts: list[np.ndarray]) -> tuple[np.ndarray, int]:
            counts = np.zeros((K, K), dtype=np.float64)
            n_pairs = 0
            for bout in bouts:
                if len(bout) < 2:
                    continue
                for i in range(len(bout) - 1):
                    counts[bout[i], bout[i + 1]] += 1
                n_pairs += len(bout) - 1
            row_sums = counts.sum(axis=1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                matrix = np.where(row_sums > 0, counts / row_sums, 0.0)
            return matrix, n_pairs

        bouts_a = self._bout_pairs_per_file(self.pop_a_df)
        bouts_b = self._bout_pairs_per_file(self.pop_b_df)
        mat_a, n_a = matrix_from_bouts(bouts_a)
        mat_b, n_b = matrix_from_bouts(bouts_b)

        frobenius = float(np.linalg.norm(mat_a - mat_b, ord="fro"))

        per_row_jsd: dict[str, float] = {}
        for i, lab in enumerate(self.all_labels):
            row_a = mat_a[i]
            row_b = mat_b[i]
            if row_a.sum() == 0 or row_b.sum() == 0:
                per_row_jsd[lab] = float("nan")
                continue
            per_row_jsd[lab] = float(jensenshannon(row_a, row_b, base=2.0) ** 2)

        interp = (
            f"Frobenius distance = {frobenius:.4f} on bout-aware "
            f"transition matrices (A: {n_a} within-bout pairs, "
            f"B: {n_b}). Per-row JSD identifies types with the most "
            f"divergent follow-on usage."
        )
        return TransitionResult(
            labels=list(self.all_labels),
            pop_a_matrix=mat_a.tolist(),
            pop_b_matrix=mat_b.tolist(),
            pop_a_n_within_pairs=int(n_a),
            pop_b_n_within_pairs=int(n_b),
            frobenius_distance=frobenius,
            per_row_jsd_bits=per_row_jsd,
            interpretation=interp,
        )

    def compare_mi_lag1(self) -> MILag1Result:
        K = len(self.all_labels)
        bouts_a = self._bout_pairs_per_file(self.pop_a_df)
        bouts_b = self._bout_pairs_per_file(self.pop_b_df)

        def mi_from_bouts(bouts: list[np.ndarray]) -> tuple[float, int, int]:
            # Reconstruct flat + ici_gap compatible with
            # mutual_information_within_bouts by concatenating with a huge
            # gap between bouts. The canonical primitive already does the
            # same thing internally via segment_into_bouts; we invoke it to
            # keep the numerical result identical to
            # scripts/analyze_sequential_structure.py.
            if not bouts:
                return 0.0, 0, 0
            codes = np.concatenate(bouts).astype(np.int64)
            # ici_gap: threshold+1 between bouts (> threshold → cut),
            # 0.0 within bouts (<= threshold → kept).
            boundary_gap = self.bout_threshold_s + 1.0
            ici = np.zeros(max(0, len(codes) - 1), dtype=np.float64)
            idx = 0
            for b_i, bout in enumerate(bouts):
                # Within this bout: len(bout) - 1 gaps at 0.0 (already).
                idx += max(0, len(bout) - 1)
                # Between this bout and the next: one boundary gap.
                if b_i < len(bouts) - 1:
                    ici[idx] = boundary_gap
                    idx += 1
            mi, n_within, n_excluded = mutual_information_within_bouts(
                codes, ici, self.bout_threshold_s, K, lag=1
            )
            return mi, n_within, n_excluded

        mi_a, n_within_a, n_excl_a = mi_from_bouts(bouts_a)
        mi_b, n_within_b, n_excl_b = mi_from_bouts(bouts_b)

        def bootstrap_mi(bouts: list[np.ndarray]) -> tuple[float, float]:
            if not bouts:
                return 0.0, 0.0
            samples = np.empty(self.n_bootstrap)
            n_bouts = len(bouts)
            for i in range(self.n_bootstrap):
                idx = self._rng.integers(0, n_bouts, size=n_bouts)
                resample = [bouts[j] for j in idx]
                mi_i, _, _ = mi_from_bouts(resample)
                samples[i] = mi_i
            return (
                float(np.percentile(samples, 2.5)),
                float(np.percentile(samples, 97.5)),
            )

        ci_a = bootstrap_mi(bouts_a)
        ci_b = bootstrap_mi(bouts_b)

        interp = (
            f"Bout-aware MI(lag=1): A={mi_a:.4f} bits [95% CI "
            f"{ci_a[0]:.4f}, {ci_a[1]:.4f}], B={mi_b:.4f} bits "
            f"[95% CI {ci_b[0]:.4f}, {ci_b[1]:.4f}]. Compare to canonical "
            f"5970 value 0.0921 bits in corpus_facts."
        )
        return MILag1Result(
            pop_a_mi_bits=mi_a,
            pop_b_mi_bits=mi_b,
            pop_a_n_within_pairs=int(n_within_a),
            pop_b_n_within_pairs=int(n_within_b),
            pop_a_n_excluded_pairs=int(n_excl_a),
            pop_b_n_excluded_pairs=int(n_excl_b),
            pop_a_ci_95=ci_a,
            pop_b_ci_95=ci_b,
            difference_bits=float(mi_a - mi_b),
            n_bootstrap=self.n_bootstrap,
            interpretation=interp,
        )

    def compare_zipf(self) -> ZipfComparisonResult:
        label_to_idx = {lab: i for i, lab in enumerate(self.all_labels)}
        codes_a = np.array(
            [label_to_idx[l] for l in self.pop_a_df[self.type_column].values],
            dtype=np.int64,
        )
        codes_b = np.array(
            [label_to_idx[l] for l in self.pop_b_df[self.type_column].values],
            dtype=np.int64,
        )

        def pop_zipf(codes: np.ndarray) -> ZipfPopResult:
            unique_types = int(np.unique(codes).size)
            if unique_types < 10:
                # Matches zipf_exponent_mle early-return; surface as a flag
                # so downstream code doesn't silently compare alpha=0.
                res = zipf_exponent_mle(codes)
                return ZipfPopResult(
                    alpha=res.alpha,
                    rank_alpha=res.rank_alpha,
                    xmin=res.xmin,
                    p_value=res.p_value,
                    n_tail=res.n_tail,
                    log_likelihood_ratio=res.log_likelihood_ratio,
                    insufficient_types=True,
                )
            res = zipf_exponent_mle(codes)
            return ZipfPopResult(
                alpha=res.alpha,
                rank_alpha=res.rank_alpha,
                xmin=res.xmin,
                p_value=res.p_value,
                n_tail=res.n_tail,
                log_likelihood_ratio=res.log_likelihood_ratio,
                insufficient_types=False,
            )

        z_a = pop_zipf(codes_a)
        z_b = pop_zipf(codes_b)
        if z_a.insufficient_types or z_b.insufficient_types:
            self._caveats.append(
                "Zipf: one or both populations have <10 unique types — "
                "alpha estimate is undefined (sentinel 0.0)."
            )
        interp = (
            f"Zipf alpha (count distribution, Clauset MLE): "
            f"A={z_a.alpha:.3f} B={z_b.alpha:.3f} "
            f"diff={z_a.alpha - z_b.alpha:+.3f}."
        )
        return ZipfComparisonResult(
            pop_a=z_a,
            pop_b=z_b,
            alpha_difference=float(z_a.alpha - z_b.alpha),
            interpretation=interp,
        )

    def compare_burstiness(self) -> BurstinessResult:
        ioi_a = self._iois_within_bouts(self.pop_a_df)
        ioi_b = self._iois_within_bouts(self.pop_b_df)

        def cv(x: np.ndarray) -> tuple[float, float, int]:
            if len(x) == 0:
                return 0.0, 0.0, 0
            mean = float(np.mean(x))
            std = float(np.std(x))
            return (std / mean if mean > 0 else 0.0), mean, int(len(x))

        cv_a, mean_a, n_a = cv(ioi_a)
        cv_b, mean_b, n_b = cv(ioi_b)

        def label(cv_val: float) -> str:
            if cv_val == 0.0:
                return "no-data"
            if cv_val < 0.8:
                return "periodic"
            if cv_val <= 1.2:
                return "poisson"
            return "bursty"

        interp = (
            f"Within-bout IOI CV: A={cv_a:.3f} ({label(cv_a)}), "
            f"B={cv_b:.3f} ({label(cv_b)}). CV<0.8 periodic, "
            f"0.8-1.2 Poisson, >1.2 bursty."
        )
        return BurstinessResult(
            pop_a_cv=cv_a,
            pop_b_cv=cv_b,
            pop_a_mean_ioi_s=mean_a,
            pop_b_mean_ioi_s=mean_b,
            pop_a_n_iois=n_a,
            pop_b_n_iois=n_b,
            interpretation=interp,
        )

    def compare_ioi_distributions(self) -> IOIDistributionResult:
        ioi_a = self._iois_within_bouts(self.pop_a_df)
        ioi_b = self._iois_within_bouts(self.pop_b_df)

        def stats(x: np.ndarray) -> tuple[float, tuple[float, float], int]:
            if len(x) == 0:
                return 0.0, (0.0, 0.0), 0
            return (
                float(np.median(x)),
                (float(np.percentile(x, 25)), float(np.percentile(x, 75))),
                int(len(x)),
            )

        med_a, iqr_a, n_a = stats(ioi_a)
        med_b, iqr_b, n_b = stats(ioi_b)

        if n_a >= 2 and n_b >= 2:
            ks = ks_2samp(ioi_a, ioi_b)
            ks_stat = float(ks.statistic)
            ks_p = float(ks.pvalue)
        else:
            ks_stat = float("nan")
            ks_p = float("nan")

        interp = (
            f"Within-bout IOI median: A={med_a*1000:.1f} ms (IQR "
            f"{iqr_a[0]*1000:.1f}-{iqr_a[1]*1000:.1f}), "
            f"B={med_b*1000:.1f} ms (IQR {iqr_b[0]*1000:.1f}-{iqr_b[1]*1000:.1f}); "
            f"KS D={ks_stat:.3f} p={ks_p:.4g}."
        )
        return IOIDistributionResult(
            pop_a_median_s=med_a,
            pop_b_median_s=med_b,
            pop_a_iqr_s=iqr_a,
            pop_b_iqr_s=iqr_b,
            ks_statistic=ks_stat,
            ks_p_value=ks_p,
            n_a=n_a,
            n_b=n_b,
            interpretation=interp,
        )

    def compare_features(self) -> FeatureComparisonResult:
        per_feature: dict[str, FeatureComparison] = {}
        max_feature: Optional[str] = None
        max_abs_d = 0.0
        skipped: list[str] = []

        for feat in self.feature_columns:
            if feat not in self.pop_a_df.columns or feat not in self.pop_b_df.columns:
                skipped.append(feat)
                continue
            a = self.pop_a_df[feat].dropna().to_numpy(dtype=np.float64)
            b = self.pop_b_df[feat].dropna().to_numpy(dtype=np.float64)
            if len(a) < 2 or len(b) < 2:
                skipped.append(feat)
                continue

            mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
            std_a, std_b = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
            med_a, med_b = float(np.median(a)), float(np.median(b))

            # Pooled SD Cohen's d.
            n_a_i, n_b_i = len(a), len(b)
            pooled_var = (
                (n_a_i - 1) * std_a ** 2 + (n_b_i - 1) * std_b ** 2
            ) / max(n_a_i + n_b_i - 2, 1)
            pooled_sd = np.sqrt(pooled_var) if pooled_var > 0 else 0.0
            d = (mean_a - mean_b) / pooled_sd if pooled_sd > 0 else 0.0

            ks = ks_2samp(a, b)
            per_feature[feat] = FeatureComparison(
                feature=feat,
                pop_a_mean=mean_a,
                pop_b_mean=mean_b,
                pop_a_std=std_a,
                pop_b_std=std_b,
                pop_a_median=med_a,
                pop_b_median=med_b,
                cohens_d=float(d),
                ks_statistic=float(ks.statistic),
                ks_p_value=float(ks.pvalue),
                n_a=n_a_i,
                n_b=n_b_i,
            )
            if abs(d) > max_abs_d:
                max_abs_d = abs(d)
                max_feature = feat

        if skipped:
            self._caveats.append(
                "Feature comparison skipped (missing or <2 values): "
                + ", ".join(skipped)
            )
        return FeatureComparisonResult(
            per_feature=per_feature,
            max_abs_cohens_d_feature=max_feature,
            max_abs_cohens_d=float(max_abs_d),
        )

    def compare_umap_overlap(
        self,
        grid_resolution: int = 80,
        subsample_n: Optional[int] = None,
    ) -> UMAPOverlapResult:
        # Prepare combined feature matrix over features present in BOTH pops.
        feat_cols = [
            f
            for f in self.feature_columns
            if f in self.pop_a_df.columns and f in self.pop_b_df.columns
        ]
        if not feat_cols:
            self._caveats.append("UMAP: no common feature columns — skipped.")
            return UMAPOverlapResult(
                overlap_coefficient=0.0,
                grid_resolution=grid_resolution,
                n_pop_a=0,
                n_pop_b=0,
                used_subsample=False,
                subsample_n=subsample_n,
                interpretation="skipped: no shared features",
            )

        a = self.pop_a_df[feat_cols].dropna().to_numpy(dtype=np.float64)
        b = self.pop_b_df[feat_cols].dropna().to_numpy(dtype=np.float64)
        used_subsample = False
        if subsample_n is not None:
            if len(a) > subsample_n:
                a = a[self._rng.choice(len(a), subsample_n, replace=False)]
                used_subsample = True
            if len(b) > subsample_n:
                b = b[self._rng.choice(len(b), subsample_n, replace=False)]
                used_subsample = True

        if len(a) < 10 or len(b) < 10:
            self._caveats.append(
                "UMAP overlap: <10 calls after NaN drop in one population — skipped."
            )
            return UMAPOverlapResult(
                overlap_coefficient=0.0,
                grid_resolution=grid_resolution,
                n_pop_a=len(a),
                n_pop_b=len(b),
                used_subsample=used_subsample,
                subsample_n=subsample_n,
                interpretation="skipped: insufficient data",
            )

        try:
            import umap  # type: ignore
        except ImportError:
            self._caveats.append(
                "UMAP package not installed — overlap set to NaN."
            )
            return UMAPOverlapResult(
                overlap_coefficient=float("nan"),
                grid_resolution=grid_resolution,
                n_pop_a=len(a),
                n_pop_b=len(b),
                used_subsample=used_subsample,
                subsample_n=subsample_n,
                interpretation="skipped: umap-learn not importable",
            )

        # Standardize before UMAP (scale-invariant projection).
        combined = np.vstack([a, b])
        mu = combined.mean(axis=0)
        sigma = combined.std(axis=0, ddof=0)
        sigma = np.where(sigma == 0, 1.0, sigma)
        combined_z = (combined - mu) / sigma

        reducer = umap.UMAP(
            n_components=2,
            random_state=self.random_state,
        )
        embedding = reducer.fit_transform(combined_z)
        emb_a = embedding[: len(a)]
        emb_b = embedding[len(a):]

        # KDE on a shared grid, then overlap coefficient = sum(min(p,q))*dA.
        x_min, x_max = float(embedding[:, 0].min()), float(embedding[:, 0].max())
        y_min, y_max = float(embedding[:, 1].min()), float(embedding[:, 1].max())
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, grid_resolution),
            np.linspace(y_min, y_max, grid_resolution),
        )
        grid = np.vstack([xx.ravel(), yy.ravel()])
        dA = (xx[0, 1] - xx[0, 0]) * (yy[1, 0] - yy[0, 0])

        kde_a = gaussian_kde(emb_a.T)
        kde_b = gaussian_kde(emb_b.T)
        p_a = kde_a(grid).reshape(xx.shape)
        p_b = kde_b(grid).reshape(xx.shape)
        # Normalize to sum to 1 on the grid (protects against edge leakage).
        p_a = p_a / (p_a.sum() * dA)
        p_b = p_b / (p_b.sum() * dA)
        overlap = float((np.minimum(p_a, p_b)).sum() * dA)

        interp = (
            f"Joint-UMAP overlap coefficient = {overlap:.3f} "
            f"(0 = disjoint, 1 = identical; grid {grid_resolution}×{grid_resolution})"
        )
        return UMAPOverlapResult(
            overlap_coefficient=overlap,
            grid_resolution=grid_resolution,
            n_pop_a=len(a),
            n_pop_b=len(b),
            used_subsample=used_subsample,
            subsample_n=subsample_n,
            interpretation=interp,
        )

    # -- Orchestrator -------------------------------------------------------

    def run_all(
        self,
        skip: Iterable[str] = (),
        umap_grid_resolution: int = 80,
        umap_subsample_n: Optional[int] = None,
    ) -> ComparisonReport:
        """Run all 10 comparisons and package into a :class:`ComparisonReport`.

        Parameters
        ----------
        skip
            Names of ``compare_*`` methods to skip (without the ``compare_``
            prefix). Useful for fast smoke runs without UMAP.
        umap_grid_resolution
            Passed to :meth:`compare_umap_overlap`.
        umap_subsample_n
            Passed to :meth:`compare_umap_overlap`.
        """
        skip_set = set(skip)

        self._print_params()

        report = ComparisonReport(
            metadata=self._build_metadata(),
        )

        def run(name: str, fn):
            if name in skip_set:
                logger.info("skip %s", name)
                return None
            logger.info("run %s", name)
            return fn()

        report.type_proportions = run("type_proportions", self.compare_type_proportions)
        report.jsd = run("jsd", self.compare_repertoires_jsd)
        report.entropy = run("entropy", self.compare_entropy)
        report.transitions = run("transitions", self.compare_transitions)
        report.mi_lag1 = run("mi_lag1", self.compare_mi_lag1)
        report.zipf = run("zipf", self.compare_zipf)
        report.burstiness = run("burstiness", self.compare_burstiness)
        report.ioi = run("ioi", self.compare_ioi_distributions)
        report.features = run("features", self.compare_features)
        report.umap_overlap = run(
            "umap_overlap",
            lambda: self.compare_umap_overlap(
                grid_resolution=umap_grid_resolution,
                subsample_n=umap_subsample_n,
            ),
        )

        # Rebuild metadata so any caveats appended during runs are captured.
        report.metadata = self._build_metadata()
        return report

    # -- Metadata / printing ------------------------------------------------

    def _build_metadata(self) -> ComparisonMetadata:
        return ComparisonMetadata(
            schema_version=SCHEMA_VERSION,
            pop_a_label=self.pop_a_label,
            pop_b_label=self.pop_b_label,
            pop_a_n_calls=int(len(self.pop_a_df)),
            pop_b_n_calls=int(len(self.pop_b_df)),
            pop_a_n_files=int(self.pop_a_df[self.file_column].nunique()),
            pop_b_n_files=int(self.pop_b_df[self.file_column].nunique()),
            bout_threshold_s=self.bout_threshold_s,
            type_column=self.type_column,
            confidence_column=self.confidence_column,
            random_state=self.random_state,
            n_bootstrap=self.n_bootstrap,
            caveats=list(self._caveats),
        )

    def _print_params(self) -> None:
        print(f"[cross_population] schema={SCHEMA_VERSION}")
        print(
            f"[cross_population] A={self.pop_a_label} "
            f"N_calls={len(self.pop_a_df)} "
            f"N_files={self.pop_a_df[self.file_column].nunique()}"
        )
        print(
            f"[cross_population] B={self.pop_b_label} "
            f"N_calls={len(self.pop_b_df)} "
            f"N_files={self.pop_b_df[self.file_column].nunique()}"
        )
        print(
            f"[cross_population] bout_threshold_s={self.bout_threshold_s} "
            f"(canonical=0.6 from corpus_facts)"
        )
        print(
            f"[cross_population] K_types={len(self.all_labels)} "
            f"n_bootstrap={self.n_bootstrap} "
            f"n_permutations={self.n_permutations} "
            f"random_state={self.random_state}"
        )
        if self.confidence_min is not None:
            print(
                f"[cross_population] confidence_min={self.confidence_min} "
                f"on {self.confidence_column}"
            )


# ---------------------------------------------------------------------------
# Report writers (free functions to keep the dataclass slim)
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot serialize {type(obj)!r}")


def _report_to_json_dict(report: ComparisonReport) -> dict:
    out: dict[str, Any] = {"metadata": asdict(report.metadata)}
    for name in (
        "type_proportions",
        "jsd",
        "entropy",
        "transitions",
        "mi_lag1",
        "zipf",
        "burstiness",
        "ioi",
        "features",
        "umap_overlap",
    ):
        val = getattr(report, name)
        if val is None:
            continue
        out[name] = asdict(val)
    out["figure_paths"] = list(report.figure_paths)
    return out


def _render_markdown(report: ComparisonReport) -> str:
    m = report.metadata
    lines = [
        f"# Cross-Population Comparison: {m.pop_a_label} vs {m.pop_b_label}",
        "",
        f"- Schema: {m.schema_version}",
        f"- A: {m.pop_a_label}  —  {m.pop_a_n_calls} calls, {m.pop_a_n_files} files",
        f"- B: {m.pop_b_label}  —  {m.pop_b_n_calls} calls, {m.pop_b_n_files} files",
        f"- Bout threshold: {m.bout_threshold_s} s (canonical)",
        f"- Random state: {m.random_state}, bootstrap N: {m.n_bootstrap}",
        "",
    ]
    if m.caveats:
        lines += ["## Caveats", ""]
        for c in m.caveats:
            lines.append(f"- {c}")
        lines.append("")

    for name, title in (
        ("type_proportions", "Type Proportions"),
        ("jsd", "JSD"),
        ("entropy", "Shannon Entropy"),
        ("transitions", "Transitions (bout-aware)"),
        ("mi_lag1", "MI at lag 1 (bout-aware)"),
        ("zipf", "Zipf"),
        ("burstiness", "Burstiness CV"),
        ("ioi", "IOI Distributions"),
        ("features", "Acoustic Features"),
        ("umap_overlap", "Joint-UMAP Overlap"),
    ):
        val = getattr(report, name)
        if val is None:
            continue
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"- **Interpretation:** {getattr(val, 'interpretation', 'n/a')}")
        lines.append("")
        lines.append("```")
        lines.append(json.dumps(asdict(val), indent=2, default=_json_default))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _render_figures(report: ComparisonReport, out_dir: Path) -> list[Path]:
    paths: list[Path] = []

    # 1. Type-proportion bar chart
    if report.type_proportions is not None:
        labels = list(report.type_proportions.pop_a_counts.keys())
        ca = np.array([report.type_proportions.pop_a_counts[l] for l in labels], float)
        cb = np.array([report.type_proportions.pop_b_counts[l] for l in labels], float)
        pa = ca / max(ca.sum(), 1)
        pb = cb / max(cb.sum(), 1)
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(labels))
        ax.bar(x - 0.18, pa, width=0.36, label=report.metadata.pop_a_label)
        ax.bar(x + 0.18, pb, width=0.36, label=report.metadata.pop_b_label)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Proportion")
        ax.set_title("Type proportions")
        ax.legend()
        p = out_dir / "type_proportions.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    # 2. Transition-matrix heatmaps side-by-side
    if report.transitions is not None:
        labels = report.transitions.labels
        mat_a = np.asarray(report.transitions.pop_a_matrix)
        mat_b = np.asarray(report.transitions.pop_b_matrix)
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, mat, pop in zip(
            axes, (mat_a, mat_b), (report.metadata.pop_a_label, report.metadata.pop_b_label)
        ):
            im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_title(pop)
            ax.set_xlabel("Next")
            ax.set_ylabel("Current")
        fig.colorbar(im, ax=axes, shrink=0.8, label="P(next | current)")
        fig.suptitle("Bout-aware transition matrices", fontsize=12)
        p = out_dir / "transitions.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    # 3. IOI distributions overlay
    if report.ioi is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axvline(report.ioi.pop_a_median_s * 1000, color="C0", linestyle="--",
                   label=f"{report.metadata.pop_a_label} median")
        ax.axvline(report.ioi.pop_b_median_s * 1000, color="C1", linestyle="--",
                   label=f"{report.metadata.pop_b_label} median")
        ax.set_xlabel("Within-bout IOI (ms)")
        ax.set_ylabel("")
        ax.set_title(
            f"IOI medians: A={report.ioi.pop_a_median_s*1000:.0f}ms, "
            f"B={report.ioi.pop_b_median_s*1000:.0f}ms "
            f"(KS p={report.ioi.ks_p_value:.2g})"
        )
        ax.legend()
        p = out_dir / "ioi_medians.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    # 4. Per-feature violin grid (Cohen's d annotated)
    if report.features is not None and report.features.per_feature:
        feats = list(report.features.per_feature.keys())
        n = len(feats)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes = np.atleast_2d(axes)
        for i, feat in enumerate(feats):
            fc = report.features.per_feature[feat]
            ax = axes[i // ncols, i % ncols]
            ax.bar(
                [report.metadata.pop_a_label, report.metadata.pop_b_label],
                [fc.pop_a_mean, fc.pop_b_mean],
                yerr=[fc.pop_a_std, fc.pop_b_std],
                capsize=4,
                color=["C0", "C1"],
            )
            ax.set_title(f"{feat}\nd={fc.cohens_d:+.2f}  KS p={fc.ks_p_value:.2g}")
        for j in range(len(feats), nrows * ncols):
            axes[j // ncols, j % ncols].axis("off")
        fig.tight_layout()
        p = out_dir / "features.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    report.figure_paths = [str(p) for p in paths]
    return paths


__all__ = [
    "CrossPopulationComparison",
    "ComparisonReport",
    "ComparisonMetadata",
    "TypeProportionResult",
    "JSDResult",
    "EntropyResult",
    "TransitionResult",
    "MILag1Result",
    "ZipfPopResult",
    "ZipfComparisonResult",
    "BurstinessResult",
    "IOIDistributionResult",
    "FeatureComparison",
    "FeatureComparisonResult",
    "UMAPOverlapResult",
    "SCHEMA_VERSION",
    "CANONICAL_BOUT_THRESHOLD_S",
    "DEFAULT_FEATURE_COLUMNS",
]
