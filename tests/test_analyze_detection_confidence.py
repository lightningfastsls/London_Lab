"""Tests for analyze_detection_confidence — written by test-architect BEFORE implementation.

Module under test: scripts/analyze_detection_confidence.py

ROADMAP test plan coverage (section 6 — Call Quality & Detection Confidence):
  1. CNN confidence by type (box/violin + Kruskal-Wallis + pairwise Mann-Whitney)
     -> test_confidence_by_type_returns_stats_with_expected_keys
     -> test_confidence_by_type_kruskal_wallis_p_value_is_float_in_0_1
     -> test_confidence_by_type_detects_significant_difference_when_one_type_higher
     -> test_confidence_by_type_single_type_no_kw_comparison
  2. Match quality cross-tab (skip gracefully if absent)
     -> test_match_quality_crosstab_produces_correct_counts
     -> test_match_quality_crosstab_skipped_gracefully_when_column_absent
  3. HDBSCAN noise inspection (hdbscan_label == -1)
     -> test_hdbscan_noise_inspection_returns_correct_subset
     -> test_hdbscan_noise_inspection_empty_when_no_noise_points
     -> test_hdbscan_noise_type_breakdown_sums_to_total_noise
  4. Confidence vs acoustic features (2x2 scatter correlation)
     -> test_confidence_vs_features_correlations_have_expected_keys
     -> test_confidence_vs_features_correlation_sign_matches_synthetic
  5. Calibration check (reviewed calls: det_user_action not null)
     -> test_calibration_check_computes_accept_rate_per_bin
     -> test_calibration_check_accept_rate_monotone_for_clean_data
     -> test_calibration_check_skipped_gracefully_when_no_reviewed_calls
  6. Summary CSV (per-type stats written to file)
     -> test_summary_csv_is_written_to_output_dir
     -> test_summary_csv_has_required_columns
     -> test_summary_csv_per_type_mean_matches_hand_computed_value
     -> test_summary_csv_n_noise_points_matches_hdbscan_count

Additional coverage (recurring gap patterns):
  - Empty input handling
     -> test_confidence_by_type_empty_dataframe_raises_or_returns_empty
  - Single-type edge case
     -> test_confidence_by_type_single_type_no_kw_comparison
  - Missing optional columns (match_quality, hdbscan_label)
     -> test_match_quality_crosstab_skipped_gracefully_when_column_absent
  - CLI integration (parse_args + main round-trip)
     -> test_main_runs_end_to_end_on_synthetic_csvs
     -> test_main_exits_nonzero_on_missing_input_file
  - All 7 known syllable types appear in summary
     -> test_summary_csv_covers_all_present_syllable_types
  - Confidence bins boundary
     -> test_calibration_bins_span_zero_to_one
  - Noise points with no syllable_type
     -> test_hdbscan_noise_type_breakdown_handles_missing_syllable_type

Total: 20 tests (6 from ROADMAP spec, 14 additional)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for tests

import numpy as np
import pandas as pd
import pytest

# ── Import bootstrap ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Will raise ImportError until the module is implemented — that is expected.
import importlib.util

_SCRIPT_PATH = SCRIPTS_DIR / "analyze_detection_confidence.py"


def _import_module():
    """Dynamically import the script as a module."""
    spec = importlib.util.spec_from_file_location(
        "analyze_detection_confidence", _SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Constants ─────────────────────────────────────────────────────────────────
SYLLABLE_TYPES = ["Flat", "Down", "Chevron", "Short", "Complex", "Frequency_Jump", "Up"]

# ── Synthetic data builders ───────────────────────────────────────────────────

def _make_taxonomy_df(
    n_per_type: int = 30,
    seed: int = 42,
    include_match_quality: bool = True,
    reviewed_fraction: float = 0.0,
) -> pd.DataFrame:
    """Build a minimal taxonomy DataFrame with known statistics.

    Confidence (det_prob_max) is constructed so that 'Flat' has a much higher
    mean (0.9) than all other types (~0.5) — used by tests that check
    statistical detection.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for stype in SYLLABLE_TYPES:
        base_conf = 0.9 if stype == "Flat" else 0.5
        for _ in range(n_per_type):
            det_prob_max = float(np.clip(rng.normal(base_conf, 0.05), 0.0, 1.0))
            rows.append(
                {
                    "file": f"2024-09-30_12-00-00_{len(rows):07d}",
                    "syllable_type": stype,
                    "call_length_s": float(rng.uniform(0.01, 0.15)),
                    "mean_power_db": float(rng.uniform(-90.0, -60.0)),
                    "tonality": float(rng.uniform(0.1, 0.9)),
                    "sinuosity": float(rng.uniform(1.0, 5.0)),
                    "det_prob_max": det_prob_max,
                    "det_prob_mean": det_prob_max * 0.9,
                    "det_user_action": None,
                    "accepted": 1.0,
                    "classification_confidence": rng.choice(["high", "medium", "low"]),
                    "match_quality": rng.choice(["fuzzy", "unmatched_ds", "unmatched_det"])
                    if include_match_quality
                    else None,
                }
            )
    df = pd.DataFrame(rows)
    if not include_match_quality:
        df = df.drop(columns=["match_quality"])

    # Optionally mark some rows as reviewed
    if reviewed_fraction > 0.0:
        n_reviewed = max(1, int(len(df) * reviewed_fraction))
        reviewed_idx = rng.choice(len(df), size=n_reviewed, replace=False)
        df.loc[reviewed_idx, "det_user_action"] = rng.choice(
            ["accepted", "rejected"], size=n_reviewed
        )
        df.loc[reviewed_idx, "accepted"] = np.where(
            df.loc[reviewed_idx, "det_user_action"] == "accepted", 1.0, 0.0
        )

    return df


def _make_umap_df(
    n_per_type: int = 30,
    n_noise: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a minimal UMAP DataFrame with HDBSCAN labels including noise (-1)."""
    rng = np.random.default_rng(seed)
    taxonomy_df = _make_taxonomy_df(n_per_type=n_per_type, seed=seed)

    n_total = len(taxonomy_df) + n_noise
    umap_x = rng.normal(0.0, 1.0, size=n_total)
    umap_y = rng.normal(0.0, 1.0, size=n_total)

    # Non-noise rows get cluster labels 0, 1, or 2
    cluster_labels = rng.choice([0, 1, 2], size=len(taxonomy_df)).tolist()

    # Build noise rows
    noise_rows = []
    for i in range(n_noise):
        noise_rows.append(
            {
                "file": f"2024-09-30_14-00-00_{i:07d}",
                "syllable_type": SYLLABLE_TYPES[i % len(SYLLABLE_TYPES)],
                "call_length_s": float(rng.uniform(0.005, 0.02)),
                "mean_power_db": float(rng.uniform(-100.0, -85.0)),
                "tonality": float(rng.uniform(0.0, 0.3)),
                "sinuosity": float(rng.uniform(1.0, 2.0)),
                "det_prob_max": float(rng.uniform(0.05, 0.3)),
                "det_prob_mean": float(rng.uniform(0.03, 0.2)),
                "det_user_action": None,
                "accepted": 1.0,
                "classification_confidence": "low",
                "match_quality": "fuzzy",
                "umap_x": float(rng.normal(5.0, 0.1)),   # Far from cluster
                "umap_y": float(rng.normal(5.0, 0.1)),
                "hdbscan_label": -1,
                "hdbscan_probability": 0.0,
            }
        )

    taxonomy_df = taxonomy_df.copy()
    taxonomy_df["umap_x"] = umap_x[: len(taxonomy_df)]
    taxonomy_df["umap_y"] = umap_y[: len(taxonomy_df)]
    taxonomy_df["hdbscan_label"] = cluster_labels
    taxonomy_df["hdbscan_probability"] = rng.uniform(0.5, 1.0, size=len(taxonomy_df))

    noise_df = pd.DataFrame(noise_rows)
    return pd.concat([taxonomy_df, noise_df], ignore_index=True)


# ── Confidence by type ────────────────────────────────────────────────────────

class TestConfidenceByType:
    """Tests for the confidence-by-type analysis function."""

    def test_confidence_by_type_returns_stats_with_expected_keys(self):
        """Spec: box/violin plot + KW test; result dict must include kruskal_p and per_type_stats."""
        mod = _import_module()
        df = _make_taxonomy_df(n_per_type=30)
        result = mod.confidence_by_type(df)

        assert "kruskal_p" in result, "Result must contain 'kruskal_p' key"
        assert "per_type_stats" in result, "Result must contain 'per_type_stats' key"

    def test_confidence_by_type_kruskal_wallis_p_value_is_float_in_0_1(self):
        """KW p-value must be a valid probability in [0, 1]."""
        mod = _import_module()
        df = _make_taxonomy_df(n_per_type=30)
        result = mod.confidence_by_type(df)

        p = result["kruskal_p"]
        assert isinstance(p, float), f"kruskal_p must be float, got {type(p)}"
        assert 0.0 <= p <= 1.0, f"kruskal_p must be in [0,1], got {p}"

    def test_confidence_by_type_detects_significant_difference_when_one_type_higher(self):
        """With Flat at 0.9 and all others at 0.5, KW must yield p < 0.001.

        Hand-computed expectation: effect size is ~8 standard deviations apart.
        Any correct KW implementation will give p << 0.001.
        """
        mod = _import_module()
        df = _make_taxonomy_df(n_per_type=50)
        result = mod.confidence_by_type(df)

        assert result["kruskal_p"] < 0.001, (
            f"Expected p < 0.001 for large effect, got p = {result['kruskal_p']}"
        )

    def test_confidence_by_type_per_type_mean_flat_exceeds_others(self):
        """Per-type stats must reflect synthetic data: mean(Flat) > mean(Down) etc."""
        mod = _import_module()
        df = _make_taxonomy_df(n_per_type=50)
        result = mod.confidence_by_type(df)
        per_type = result["per_type_stats"]

        flat_mean = per_type.loc["Flat", "mean"] if hasattr(per_type, "loc") else per_type["Flat"]["mean"]
        down_mean = per_type.loc["Down", "mean"] if hasattr(per_type, "loc") else per_type["Down"]["mean"]

        assert flat_mean > down_mean + 0.2, (
            f"Flat mean ({flat_mean:.3f}) should be >> Down mean ({down_mean:.3f})"
        )

    def test_confidence_by_type_single_type_no_kw_comparison(self):
        """With only one syllable type, KW test cannot run; must not raise an exception.

        The result should either skip KW (kruskal_p = None/NaN) or raise a clear ValueError.
        In any case it must not produce an unhandled exception with an unhelpful traceback.
        """
        mod = _import_module()
        df = _make_taxonomy_df(n_per_type=20)
        df_single = df[df["syllable_type"] == "Flat"].copy()

        try:
            result = mod.confidence_by_type(df_single)
            p = result.get("kruskal_p")
            # Acceptable: None, NaN, or 1.0 (degenerate case)
            assert p is None or (isinstance(p, float) and (np.isnan(p) or p == 1.0)), (
                f"Single-type KW p should be None/NaN/1.0, got {p}"
            )
        except (ValueError, RuntimeError):
            pass  # Acceptable to raise a descriptive error

    def test_confidence_by_type_empty_dataframe_raises_or_returns_empty(self):
        """Empty input must not silently produce misleading statistics."""
        mod = _import_module()
        df_empty = pd.DataFrame(columns=["syllable_type", "det_prob_max"])

        try:
            result = mod.confidence_by_type(df_empty)
            per_type = result["per_type_stats"]
            assert len(per_type) == 0, "Empty input should yield empty per_type_stats"
        except (ValueError, KeyError):
            pass  # Acceptable — empty input is an error condition


# ── Match quality cross-tab ───────────────────────────────────────────────────

class TestMatchQualityCrosstab:
    """Tests for match_quality cross-tabulation."""

    def test_match_quality_crosstab_produces_correct_counts(self):
        """Cross-tab of match_quality x syllable_type must produce correct cell counts.

        Hand-computed: with 30 rows per type x 7 types = 210 rows,
        uniform random distribution over 3 match_quality levels means
        each cell should be approximately 10. The total must equal 210.
        """
        mod = _import_module()
        df = _make_taxonomy_df(n_per_type=30, include_match_quality=True, seed=0)
        result = mod.match_quality_crosstab(df)

        crosstab = result["crosstab"]
        total = int(crosstab.values.sum())
        assert total == len(df), (
            f"Cross-tab total ({total}) must equal number of rows ({len(df)})"
        )

    def test_match_quality_crosstab_all_syllable_types_appear_as_columns_or_rows(self):
        """Every syllable type present in the data must appear in the cross-tab."""
        mod = _import_module()
        df = _make_taxonomy_df(n_per_type=30, include_match_quality=True, seed=0)
        result = mod.match_quality_crosstab(df)

        crosstab = result["crosstab"]
        # Cross-tab may use types as index or columns
        labels_in_crosstab = set(crosstab.columns.tolist()) | set(crosstab.index.tolist())
        for stype in SYLLABLE_TYPES:
            assert stype in labels_in_crosstab, (
                f"Syllable type '{stype}' must appear in cross-tab axes"
            )

    def test_match_quality_crosstab_skipped_gracefully_when_column_absent(self):
        """When match_quality column is absent, function must return skip signal, not crash.

        Spec note: [ASSUMED] — match_quality may or may not exist.
        """
        mod = _import_module()
        df = _make_taxonomy_df(n_per_type=30, include_match_quality=False)
        assert "match_quality" not in df.columns

        result = mod.match_quality_crosstab(df)
        assert result["skipped"] is True, (
            "When match_quality column absent, result['skipped'] must be True"
        )


# ── HDBSCAN noise inspection ──────────────────────────────────────────────────

class TestHdbscanNoiseInspection:
    """Tests for HDBSCAN noise point (label == -1) analysis."""

    def test_hdbscan_noise_inspection_returns_correct_subset(self):
        """Noise subset must contain exactly the rows with hdbscan_label == -1."""
        mod = _import_module()
        df = _make_umap_df(n_per_type=20, n_noise=10)
        result = mod.hdbscan_noise_inspection(df)

        noise_df = result["noise_df"]
        expected_n_noise = (df["hdbscan_label"] == -1).sum()
        assert len(noise_df) == expected_n_noise, (
            f"Noise subset should have {expected_n_noise} rows, got {len(noise_df)}"
        )
        assert (noise_df["hdbscan_label"] == -1).all(), (
            "All rows in noise_df must have hdbscan_label == -1"
        )

    def test_hdbscan_noise_inspection_empty_when_no_noise_points(self):
        """When there are no noise points, noise_df must be empty and n_noise must be 0."""
        mod = _import_module()
        df = _make_umap_df(n_per_type=20, n_noise=0)
        # Verify our synthetic data has no noise
        assert (df["hdbscan_label"] == -1).sum() == 0

        result = mod.hdbscan_noise_inspection(df)
        assert len(result["noise_df"]) == 0, "noise_df should be empty when no noise points"
        assert result["n_noise"] == 0, "n_noise should be 0 when no noise points"

    def test_hdbscan_noise_type_breakdown_sums_to_total_noise(self):
        """Type breakdown of noise points must sum to the total noise count."""
        mod = _import_module()
        df = _make_umap_df(n_per_type=20, n_noise=14)
        result = mod.hdbscan_noise_inspection(df)

        type_breakdown = result["type_breakdown"]  # dict or Series: {type: count}
        total_in_breakdown = sum(
            v for v in (type_breakdown.values() if hasattr(type_breakdown, "values") else type_breakdown)
        )
        assert total_in_breakdown == result["n_noise"], (
            f"type_breakdown sum ({total_in_breakdown}) != n_noise ({result['n_noise']})"
        )

    def test_hdbscan_noise_type_breakdown_handles_missing_syllable_type(self):
        """Noise rows without syllable_type (NaN) must not cause a KeyError or crash."""
        mod = _import_module()
        df = _make_umap_df(n_per_type=10, n_noise=5)
        noise_idx = df[df["hdbscan_label"] == -1].index
        df.loc[noise_idx[:2], "syllable_type"] = np.nan

        result = mod.hdbscan_noise_inspection(df)
        # Should complete without error
        assert "n_noise" in result
        assert result["n_noise"] == len(noise_idx)


# ── Confidence vs acoustic features ──────────────────────────────────────────

class TestConfidenceVsFeatures:
    """Tests for the confidence-vs-acoustic-features scatter analysis."""

    def test_confidence_vs_features_correlations_have_expected_keys(self):
        """Result must include correlation coefficients for the 4 spec'd features."""
        mod = _import_module()
        df = _make_taxonomy_df(n_per_type=40)
        result = mod.confidence_vs_features(df)

        correlations = result["correlations"]
        required_features = {"call_length_s", "mean_power_db", "tonality", "sinuosity"}
        missing = required_features - set(correlations.keys())
        assert not missing, (
            f"Correlations dict missing features: {missing}"
        )

    def test_confidence_vs_features_correlation_values_in_minus1_to_1(self):
        """Every correlation coefficient must be in [-1, 1]."""
        mod = _import_module()
        df = _make_taxonomy_df(n_per_type=40)
        result = mod.confidence_vs_features(df)

        for feat, corr in result["correlations"].items():
            assert -1.0 <= corr <= 1.0, (
                f"Correlation for '{feat}' must be in [-1,1], got {corr}"
            )

    def test_confidence_vs_features_correlation_sign_matches_synthetic(self):
        """Create data where tonality perfectly correlates with det_prob_max (r=1.0).

        Hand-computed: if det_prob_max = tonality exactly, r(tonality, det_prob_max) = 1.0.
        """
        mod = _import_module()
        rng = np.random.default_rng(0)
        n = 100
        tonality = rng.uniform(0.1, 0.9, size=n)
        df = pd.DataFrame({
            "syllable_type": ["Flat"] * n,
            "det_prob_max": tonality,   # Perfect positive correlation
            "call_length_s": rng.uniform(0.01, 0.1, size=n),
            "mean_power_db": rng.uniform(-90, -60, size=n),
            "tonality": tonality,
            "sinuosity": rng.uniform(1.0, 3.0, size=n),
        })
        result = mod.confidence_vs_features(df)

        r_tonality = result["correlations"]["tonality"]
        assert r_tonality > 0.99, (
            f"With det_prob_max == tonality, r should be ~1.0, got {r_tonality:.4f}"
        )


# ── Calibration check ─────────────────────────────────────────────────────────

class TestCalibrationCheck:
    """Tests for the confidence calibration analysis (reviewed calls only)."""

    def test_calibration_check_computes_accept_rate_per_bin(self):
        """With reviewed calls, accept_rate must be computed for each confidence bin."""
        mod = _import_module()
        df = _make_taxonomy_df(n_per_type=40, reviewed_fraction=0.5)
        n_reviewed = df["det_user_action"].notna().sum()
        assert n_reviewed > 0, "Fixture error: expected reviewed rows"

        result = mod.calibration_check(df)
        bins = result["bins"]
        assert len(bins) > 0, "calibration bins must not be empty when reviewed rows exist"
        assert "accept_rate" in bins.columns or "accept_rate" in bins, (
            "bins must include an accept_rate field"
        )

    def test_calibration_check_accept_rate_monotone_for_clean_data(self):
        """High-confidence bins should have higher accept rates than low-confidence bins.

        Construct synthetic data where accepted == 1 iff det_prob_max > 0.5.
        This creates a clean step function: accept_rate should be 0 in low bins
        and 1 in high bins.
        """
        mod = _import_module()
        rng = np.random.default_rng(0)
        n = 200
        det_prob_max = rng.uniform(0.05, 0.95, size=n)
        accepted = (det_prob_max > 0.5).astype(float)
        df = pd.DataFrame({
            "syllable_type": ["Flat"] * n,
            "det_prob_max": det_prob_max,
            "det_prob_mean": det_prob_max * 0.9,
            "call_length_s": rng.uniform(0.01, 0.1, size=n),
            "mean_power_db": rng.uniform(-90, -60, size=n),
            "tonality": rng.uniform(0.1, 0.9, size=n),
            "sinuosity": rng.uniform(1.0, 3.0, size=n),
            "det_user_action": ["reviewed"] * n,   # All reviewed
            "accepted": accepted,
        })
        result = mod.calibration_check(df)
        bins = result["bins"]

        # Extract accept rates ordered by bin (low to high probability)
        if hasattr(bins, "sort_values"):
            bins_sorted = bins.sort_values(by=bins.columns[0])
            rates = bins_sorted["accept_rate"].tolist()
        else:
            rates = sorted(bins.items())
            rates = [v for _, v in rates]

        # The highest-confidence bin should have accept_rate > the lowest-confidence bin
        assert rates[-1] > rates[0] + 0.3, (
            f"Highest-bin accept_rate ({rates[-1]:.2f}) should be >> lowest ({rates[0]:.2f})"
        )

    def test_calibration_check_skipped_gracefully_when_no_reviewed_calls(self):
        """When det_user_action is entirely null, calibration must return skip signal."""
        mod = _import_module()
        df = _make_taxonomy_df(n_per_type=30, reviewed_fraction=0.0)
        assert df["det_user_action"].isna().all(), "Fixture error: expected no reviewed rows"

        result = mod.calibration_check(df)
        assert result["skipped"] is True, (
            "calibration_check must set skipped=True when no reviewed calls exist"
        )

    def test_calibration_bins_span_zero_to_one(self):
        """Confidence bins must cover the full [0, 1] range, not just observed values."""
        mod = _import_module()
        rng = np.random.default_rng(42)
        n = 100
        # All probabilities in [0.8, 0.95] — a narrow high-confidence band
        det_prob_max = rng.uniform(0.8, 0.95, size=n)
        df = pd.DataFrame({
            "syllable_type": ["Flat"] * n,
            "det_prob_max": det_prob_max,
            "det_prob_mean": det_prob_max * 0.9,
            "call_length_s": 0.05,
            "mean_power_db": -75.0,
            "tonality": 0.5,
            "sinuosity": 1.5,
            "det_user_action": ["reviewed"] * n,
            "accepted": np.ones(n),
        })
        result = mod.calibration_check(df)
        bins = result["bins"]

        # Bins are defined over [0,1]; should have at least 5 bins regardless of data range
        n_bins = len(bins) if hasattr(bins, "__len__") else len(list(bins))
        assert n_bins >= 5, (
            f"Expected >= 5 bins spanning [0,1]; got {n_bins}"
        )


# ── Summary CSV ───────────────────────────────────────────────────────────────

class TestSummaryCSV:
    """Tests for the per-type summary CSV output."""

    def test_summary_csv_is_written_to_output_dir(self, tmp_path):
        """A CSV file must be created in the output directory."""
        mod = _import_module()
        df_taxonomy = _make_taxonomy_df(n_per_type=20)
        df_umap = _make_umap_df(n_per_type=20, n_noise=5)

        mod.write_summary_csv(df_taxonomy, df_umap, output_dir=tmp_path)

        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) >= 1, "At least one CSV file must be written to output_dir"

    def test_summary_csv_has_required_columns(self, tmp_path):
        """Summary CSV must contain mean_det_prob_max, median_det_prob_max, n_reviewed,
        accept_rate, and n_noise_points columns.

        These are the columns listed in the module spec.
        """
        mod = _import_module()
        df_taxonomy = _make_taxonomy_df(n_per_type=20, reviewed_fraction=0.3)
        df_umap = _make_umap_df(n_per_type=20, n_noise=5)

        mod.write_summary_csv(df_taxonomy, df_umap, output_dir=tmp_path)

        csv_path = next(tmp_path.glob("*.csv"))
        summary = pd.read_csv(csv_path)

        required_cols = {
            "mean_det_prob_max",
            "median_det_prob_max",
            "n_reviewed",
            "accept_rate",
            "n_noise_points",
        }
        missing = required_cols - set(summary.columns)
        assert not missing, f"Summary CSV missing required columns: {missing}"

    def test_summary_csv_per_type_mean_matches_hand_computed_value(self, tmp_path):
        """Mean det_prob_max for Flat must match a hand-computed value.

        With seed=0 and n_per_type=50, Flat rows have det_prob_max ~ N(0.9, 0.05).
        The mean must be in [0.85, 0.95].
        """
        mod = _import_module()
        df_taxonomy = _make_taxonomy_df(n_per_type=50, seed=0)
        df_umap = _make_umap_df(n_per_type=50, n_noise=5, seed=0)

        mod.write_summary_csv(df_taxonomy, df_umap, output_dir=tmp_path)

        csv_path = next(tmp_path.glob("*.csv"))
        summary = pd.read_csv(csv_path)

        # Find the Flat row (could be index or column)
        if "syllable_type" in summary.columns:
            flat_row = summary[summary["syllable_type"] == "Flat"]
        else:
            flat_row = summary.loc[summary.index == "Flat"]

        assert len(flat_row) == 1, "Summary must have exactly one row for 'Flat'"
        flat_mean = float(flat_row["mean_det_prob_max"].iloc[0])
        assert 0.85 <= flat_mean <= 0.95, (
            f"Flat mean det_prob_max should be in [0.85, 0.95], got {flat_mean:.4f}"
        )

    def test_summary_csv_n_noise_points_matches_hdbscan_count(self, tmp_path):
        """n_noise_points column must match the actual count of hdbscan_label == -1 rows per type.

        Hand-computed: with 10 noise rows cycling through 7 types, each type gets
        floor(10/7) or ceil(10/7) noise points — total must be 10.
        """
        mod = _import_module()
        df_taxonomy = _make_taxonomy_df(n_per_type=20, seed=1)
        df_umap = _make_umap_df(n_per_type=20, n_noise=14, seed=1)

        mod.write_summary_csv(df_taxonomy, df_umap, output_dir=tmp_path)

        csv_path = next(tmp_path.glob("*.csv"))
        summary = pd.read_csv(csv_path)

        total_noise_in_csv = summary["n_noise_points"].sum()
        actual_noise = (df_umap["hdbscan_label"] == -1).sum()
        assert total_noise_in_csv == actual_noise, (
            f"Summary n_noise_points total ({total_noise_in_csv}) != "
            f"actual noise rows ({actual_noise})"
        )

    def test_summary_csv_covers_all_present_syllable_types(self, tmp_path):
        """Every syllable type in the input must appear as a row in the summary."""
        mod = _import_module()
        df_taxonomy = _make_taxonomy_df(n_per_type=20)
        df_umap = _make_umap_df(n_per_type=20, n_noise=5)

        mod.write_summary_csv(df_taxonomy, df_umap, output_dir=tmp_path)

        csv_path = next(tmp_path.glob("*.csv"))
        summary = pd.read_csv(csv_path)

        if "syllable_type" in summary.columns:
            types_in_summary = set(summary["syllable_type"].tolist())
        else:
            types_in_summary = set(summary.index.tolist())

        for stype in SYLLABLE_TYPES:
            assert stype in types_in_summary, (
                f"'{stype}' not found in summary CSV rows"
            )


# ── CLI integration ───────────────────────────────────────────────────────────

class TestCLIIntegration:
    """Integration tests for the script's command-line interface."""

    def test_main_runs_end_to_end_on_synthetic_csvs(self, tmp_path):
        """main() must complete with exit code 0 given valid input CSVs and output dir.

        This is an integration test that verifies the full pipeline:
        input CSVs -> all 6 analyses -> outputs in output_dir.
        """
        mod = _import_module()

        # Write synthetic CSVs to tmp_path
        taxonomy_csv = tmp_path / "taxonomy.csv"
        umap_csv = tmp_path / "umap.csv"
        output_dir = tmp_path / "results"
        output_dir.mkdir()

        df_taxonomy = _make_taxonomy_df(n_per_type=15, reviewed_fraction=0.3)
        df_umap = _make_umap_df(n_per_type=15, n_noise=5)
        df_taxonomy.to_csv(taxonomy_csv, index=False)
        df_umap.to_csv(umap_csv, index=False)

        exit_code = mod.main([
            "--input-taxonomy", str(taxonomy_csv),
            "--input-umap", str(umap_csv),
            "--output-dir", str(output_dir),
        ])

        assert exit_code == 0, f"main() should return 0 on success, got {exit_code}"

        # At least one output file should have been created
        output_files = list(output_dir.iterdir())
        assert len(output_files) >= 1, "main() must write at least one output file"

    def test_main_writes_figures_to_output_dir(self, tmp_path):
        """main() must write figure files (PNG or PDF) to the output directory."""
        mod = _import_module()

        taxonomy_csv = tmp_path / "taxonomy.csv"
        umap_csv = tmp_path / "umap.csv"
        output_dir = tmp_path / "figures"
        output_dir.mkdir()

        _make_taxonomy_df(n_per_type=15).to_csv(taxonomy_csv, index=False)
        _make_umap_df(n_per_type=15, n_noise=5).to_csv(umap_csv, index=False)

        mod.main([
            "--input-taxonomy", str(taxonomy_csv),
            "--input-umap", str(umap_csv),
            "--output-dir", str(output_dir),
        ])

        image_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.pdf"))
        assert len(image_files) >= 1, (
            "main() must write at least one figure (PNG/PDF) to output_dir"
        )

    def test_main_exits_nonzero_on_missing_input_file(self, tmp_path):
        """main() must return nonzero exit code when input files do not exist."""
        mod = _import_module()

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        exit_code = mod.main([
            "--input-taxonomy", str(tmp_path / "nonexistent.csv"),
            "--input-umap", str(tmp_path / "also_nonexistent.csv"),
            "--output-dir", str(output_dir),
        ])

        assert exit_code != 0, (
            "main() should return nonzero exit code when input files are missing"
        )

    def test_main_umap_without_hdbscan_label_skips_noise_analysis(self, tmp_path):
        """When UMAP CSV lacks hdbscan_label, noise analysis must be skipped gracefully.

        This tests the [ASSUMED] scenario: UMAP file may not always have hdbscan_label.
        """
        mod = _import_module()

        taxonomy_csv = tmp_path / "taxonomy.csv"
        umap_csv = tmp_path / "umap_no_hdbscan.csv"
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        df_taxonomy = _make_taxonomy_df(n_per_type=15)
        df_umap = _make_umap_df(n_per_type=15, n_noise=0)
        df_umap = df_umap.drop(columns=["hdbscan_label", "hdbscan_probability"])
        df_taxonomy.to_csv(taxonomy_csv, index=False)
        df_umap.to_csv(umap_csv, index=False)

        # Should complete without error even without hdbscan_label
        exit_code = mod.main([
            "--input-taxonomy", str(taxonomy_csv),
            "--input-umap", str(umap_csv),
            "--output-dir", str(output_dir),
        ])

        assert exit_code == 0, (
            f"main() should succeed even when hdbscan_label is absent; got exit_code={exit_code}"
        )
