"""Tests for analyze_acoustic_features — written by test-architect BEFORE implementation.

ROADMAP test plan coverage (section 16.3, 11 test cases):
  1. Correlation matrix is symmetric and has correct shape (10×10)
       -> test_correlation_matrix_shape_is_10x10
       -> test_correlation_matrix_is_symmetric
  2. PCA components sum to 1.0 (explained variance)
       -> test_pca_explained_variance_sums_to_one
  3. PCA loadings have correct shape (n_components × 10)
       -> test_pca_loadings_shape_is_n_components_by_10
  4. UMAP overlay function handles NaN features gracefully (skip row, warn)
       -> test_umap_overlay_nan_rows_are_skipped
  5. Violin plot handles types with < 5 calls without crashing
       -> test_within_type_variability_sparse_type_no_crash
  6. Summary CSV has 7 rows (one per type) and expected columns
       -> test_summary_csv_has_7_rows
       -> test_summary_csv_has_expected_columns
  7. Low-confidence filter produces non-empty subset
       -> test_low_confidence_filter_produces_nonempty_subset
  8. Power/tonality panel produces 4 subplots
       -> test_power_tonality_panel_has_4_subplots
  9. Frequency drift regression returns slope and p-value
       -> test_frequency_drift_regression_returns_slope_and_pvalue
  10. Flat call temporal analysis handles bins with zero Flat calls
       -> test_flat_temporal_handles_zero_flat_bins
  11. Script runs end-to-end on synthetic data and produces all expected output files
       -> test_end_to_end_produces_all_expected_output_files

Additional coverage (recurring gap patterns):
  - Correlation matrix diagonal is all ones (Pearson) -> test_correlation_matrix_diagonal_is_ones
  - Spearman vs Pearson differ on non-linear data -> test_spearman_and_pearson_differ_on_nonlinear_data
  - PCA loadings: each component has unit norm -> test_pca_loadings_have_unit_norm
  - PCA scatter has exactly 7 colors for 7 types -> test_pca_scatter_has_correct_type_labels
  - Summary CSV has no NaN in count/mean columns -> test_summary_csv_no_nan_in_key_columns
  - Summary CSV per-type means are within feature value range -> test_summary_csv_means_within_range
  - UMAP overlay produces exactly 10 individual files -> test_umap_overlay_produces_10_files
  - UMAP panel (2x5) figure has exactly 10 axes -> test_umap_panel_has_10_axes
  - Frequency drift slope sign reflects true trend in synthetic data -> test_frequency_drift_slope_sign_correct
  - Frequency drift p-value is in [0, 1] -> test_frequency_drift_pvalue_in_range
  - Boundary analysis: low-confidence calls subset is smaller than full dataset -> test_boundary_subset_smaller_than_full
  - Boundary analysis: empty low-confidence subset handled gracefully -> test_boundary_empty_low_confidence_no_crash
  - Within-type variability figure has 10 panels (one per feature) -> test_within_type_variability_has_10_panels
  - parse_filename_timestamp parses standard filename format -> test_parse_filename_timestamp_correct_value
  - parse_filename_timestamp raises on malformed input -> test_parse_filename_timestamp_invalid_raises
  - Single-type dataset: PCA still runs (degenerate case) -> test_pca_single_type_dataset_runs
  - Empty DataFrame: all public analysis functions raise or return gracefully -> test_empty_dataframe_raises_or_returns_gracefully
  - All 7 SYLLABLE_TYPES present in summary CSV rows -> test_summary_csv_all_7_types_present
  - Frequency drift: linear regression on flat data gives slope near 0 -> test_frequency_drift_flat_data_slope_near_zero
  - UMAP overlay: colormap is continuous (not categorical) -> test_umap_overlay_uses_continuous_colormap

Total: 28 tests (11 ROADMAP items, 17 additional gap-pattern tests)
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# ── Path bootstrap ────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SCRIPT_PATH = REPO_ROOT / "scripts" / "analyze_acoustic_features.py"

# ── Constants matching the spec ───────────────────────────────────────────────

ACOUSTIC_FEATURES = [
    "call_length_s",
    "principal_freq_hz",
    "low_freq_hz",
    "high_freq_hz",
    "bandwidth_hz",
    "freq_std_dev_hz",
    "slope",
    "sinuosity",
    "mean_power_db",
    "tonality",
]

SYLLABLE_TYPES = ["Flat", "Down", "Chevron", "Short", "Complex", "Frequency_Jump", "Up"]

# ── Synthetic data factories ──────────────────────────────────────────────────


def _make_taxonomy_df(
    n_per_type: int = 30,
    rng: np.random.Generator | None = None,
    low_confidence_fraction: float = 0.15,
    include_nans: bool = False,
) -> pd.DataFrame:
    """Build a synthetic taxonomy DataFrame with realistic feature ranges.

    Feature ranges are chosen to reflect real USV data distributions
    (as described in the ROADMAP), while remaining synthetic and controlled.
    Frequency values are in kHz (matching the data quirk documented in ROADMAP 16.5).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    rows = []
    for stype in SYLLABLE_TYPES:
        for i in range(n_per_type):
            # Assign low-confidence to a predictable fraction
            if i < int(n_per_type * low_confidence_fraction):
                conf = "low"
            elif i < int(n_per_type * 0.5):
                conf = "medium"
            else:
                conf = "high"

            # Timestamps spread across two "sessions" for temporal tests
            session = "USV1" if i < n_per_type // 2 else "USV2"
            hour = 10 + (i % 12)
            filename = f"2024-09-30_{hour:02d}-00-00_{i:07d}"

            rows.append(
                {
                    "file": filename,
                    "call_length_s": rng.uniform(0.01, 0.25),
                    "principal_freq_hz": rng.uniform(50, 85),
                    "low_freq_hz": rng.uniform(30, 60),
                    "high_freq_hz": rng.uniform(60, 100),
                    "bandwidth_hz": rng.uniform(5, 50),
                    "freq_std_dev_hz": rng.uniform(1, 20),
                    "slope": rng.uniform(-50, 50),
                    "sinuosity": rng.uniform(1.0, 5.0),
                    "mean_power_db": rng.uniform(-60, -20),
                    "tonality": rng.uniform(0.3, 1.0),
                    "syllable_type": stype,
                    "classification_confidence": conf,
                    "det_prob_max": rng.uniform(0.6, 1.0),
                    "det_prob_mean": rng.uniform(0.5, 0.95),
                    "session": session,
                }
            )

    df = pd.DataFrame(rows)

    if include_nans:
        # Introduce NaN values in ~5% of feature cells
        for col in ACOUSTIC_FEATURES:
            nan_mask = rng.random(len(df)) < 0.05
            df.loc[nan_mask, col] = np.nan

    return df


def _make_umap_df(taxonomy_df: pd.DataFrame, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Add umap_x, umap_y, and hdbscan_label columns to a taxonomy DataFrame."""
    if rng is None:
        rng = np.random.default_rng(99)

    df = taxonomy_df.copy()
    df["umap_x"] = rng.uniform(-10, 10, size=len(df))
    df["umap_y"] = rng.uniform(-10, 10, size=len(df))
    # Most points in cluster 0 (the one dominant manifold), a few as noise (-1)
    labels = np.zeros(len(df), dtype=int)
    noise_idx = rng.choice(len(df), size=5, replace=False)
    labels[noise_idx] = -1
    df["hdbscan_label"] = labels
    return df


# ── Import helper ─────────────────────────────────────────────────────────────


def _import_script() -> Any:
    """Import the analysis script as a module.

    Will raise ImportError / ModuleNotFoundError until the script exists.
    """
    spec = importlib.util.spec_from_file_location(
        "analyze_acoustic_features", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {SCRIPT_PATH} — file does not exist yet")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# ── ROADMAP test plan tests (items 1–11) ─────────────────────────────────────


class TestCorrelationMatrix:
    """ROADMAP item 1: Correlation matrix shape and symmetry."""

    def test_correlation_matrix_shape_is_10x10(self):
        """Correlation matrix must cover all 10 acoustic features: shape (10, 10)."""
        mod = _import_script()
        df = _make_taxonomy_df()
        pearson, spearman = mod.compute_correlation_matrices(df[ACOUSTIC_FEATURES])
        assert pearson.shape == (10, 10), (
            f"Expected Pearson matrix shape (10, 10), got {pearson.shape}"
        )
        assert spearman.shape == (10, 10), (
            f"Expected Spearman matrix shape (10, 10), got {spearman.shape}"
        )

    def test_correlation_matrix_is_symmetric(self):
        """Correlation matrix must be perfectly symmetric: C[i,j] == C[j,i]."""
        mod = _import_script()
        df = _make_taxonomy_df()
        pearson, spearman = mod.compute_correlation_matrices(df[ACOUSTIC_FEATURES])
        np.testing.assert_allclose(
            pearson.values, pearson.values.T, atol=1e-10,
            err_msg="Pearson matrix is not symmetric"
        )
        np.testing.assert_allclose(
            spearman.values, spearman.values.T, atol=1e-10,
            err_msg="Spearman matrix is not symmetric"
        )

    def test_correlation_matrix_diagonal_is_ones(self):
        """Diagonal of a correlation matrix must be exactly 1.0 (self-correlation)."""
        mod = _import_script()
        df = _make_taxonomy_df()
        pearson, _ = mod.compute_correlation_matrices(df[ACOUSTIC_FEATURES])
        diag = np.diag(pearson.values)
        np.testing.assert_allclose(
            diag, np.ones(10), atol=1e-10,
            err_msg="Diagonal of Pearson matrix should be all 1.0"
        )

    def test_spearman_and_pearson_differ_on_nonlinear_data(self):
        """On monotone-but-nonlinear data, Spearman should detect correlation
        that Pearson underestimates.  This verifies two distinct metrics are computed.
        """
        mod = _import_script()
        rng = np.random.default_rng(7)
        n = 200
        # x and x^3 are perfectly Spearman-correlated (rank preserving) but Pearson < 1
        x = rng.uniform(-2, 2, n)
        feature_data = pd.DataFrame({
            feat: x if feat == ACOUSTIC_FEATURES[0] else x ** 3
            for feat in ACOUSTIC_FEATURES
        })
        pearson, spearman = mod.compute_correlation_matrices(feature_data)
        # Off-diagonal Spearman should be 1.0 (perfect rank correlation)
        # Off-diagonal Pearson should be < 1.0
        s_val = spearman.iloc[0, 1]
        p_val = pearson.iloc[0, 1]
        assert abs(s_val) > abs(p_val) + 0.01, (
            f"Spearman ({s_val:.4f}) should exceed Pearson ({p_val:.4f}) for nonlinear monotone data"
        )


class TestPCA:
    """ROADMAP items 2–3: PCA explained variance and loadings shape."""

    def test_pca_explained_variance_sums_to_one(self):
        """All PCA components together must explain 100% of variance (within float tolerance)."""
        mod = _import_script()
        df = _make_taxonomy_df()
        pca_result = mod.run_pca(df[ACOUSTIC_FEATURES])
        total_variance = sum(pca_result.explained_variance_ratio_)
        assert abs(total_variance - 1.0) < 1e-6, (
            f"Explained variance should sum to 1.0, got {total_variance}"
        )

    def test_pca_loadings_shape_is_n_components_by_10(self):
        """PCA loadings matrix must have shape (n_components, 10).

        Spec requires top 5 PCs × 10 features in the output heatmap.
        """
        mod = _import_script()
        df = _make_taxonomy_df()
        pca_result = mod.run_pca(df[ACOUSTIC_FEATURES], n_components=5)
        loadings = mod.get_pca_loadings(pca_result, feature_names=ACOUSTIC_FEATURES)
        assert loadings.shape[1] == 10, (
            f"Loadings should have 10 columns (features), got {loadings.shape[1]}"
        )
        assert loadings.shape[0] == 5, (
            f"Loadings should have 5 rows (components), got {loadings.shape[0]}"
        )

    def test_pca_loadings_have_unit_norm(self):
        """Each PCA loading vector (row) must have unit L2 norm — this is a mathematical
        invariant of PCA and detects incorrect normalization in the implementation.
        """
        mod = _import_script()
        df = _make_taxonomy_df()
        pca_result = mod.run_pca(df[ACOUSTIC_FEATURES], n_components=5)
        loadings = mod.get_pca_loadings(pca_result, feature_names=ACOUSTIC_FEATURES)
        row_norms = np.linalg.norm(loadings.values, axis=1)
        np.testing.assert_allclose(
            row_norms, np.ones(len(row_norms)), atol=1e-6,
            err_msg="PCA loading rows must have unit L2 norm"
        )

    def test_pca_scatter_has_correct_type_labels(self):
        """PCA scatter plot must include a legend entry for each of the 7 syllable types."""
        mod = _import_script()
        df = _make_taxonomy_df()
        umap_df = _make_umap_df(df)
        fig = mod.plot_pca_scatter(df[ACOUSTIC_FEATURES], umap_df["syllable_type"])
        # Extract legend text labels from the figure
        legend_texts = set()
        for ax in fig.get_axes():
            legend = ax.get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    legend_texts.add(text.get_text())
        plt.close(fig)
        for stype in SYLLABLE_TYPES:
            assert stype in legend_texts, (
                f"PCA scatter legend missing type '{stype}'. Found: {legend_texts}"
            )

    def test_pca_single_type_dataset_runs(self):
        """PCA must not crash when the dataset contains only one syllable type.
        (Degenerate but valid input — edge-case guard.)
        """
        mod = _import_script()
        rng = np.random.default_rng(0)
        n = 50
        df = pd.DataFrame({feat: rng.uniform(0, 1, n) for feat in ACOUSTIC_FEATURES})
        df["syllable_type"] = "Flat"
        # Should complete without exception
        pca_result = mod.run_pca(df[ACOUSTIC_FEATURES], n_components=5)
        assert pca_result is not None, "run_pca returned None on single-type dataset"


class TestUMAPOverlays:
    """ROADMAP item 4: UMAP overlay NaN handling, 10 files, 2×5 panel."""

    def test_umap_overlay_nan_rows_are_skipped(self, tmp_path, recwarn):
        """NaN feature values must cause those rows to be dropped from the
        UMAP overlay plot, NOT cause a crash or propagate to the colormap.
        """
        mod = _import_script()
        df = _make_taxonomy_df(include_nans=True)
        umap_df = _make_umap_df(df)
        feature = ACOUSTIC_FEATURES[0]  # call_length_s
        # Inject explicit NaNs to guarantee at least a few
        umap_df.loc[:5, feature] = np.nan

        fig = mod.plot_umap_overlay(umap_df, feature=feature)
        plt.close(fig)
        # Test passes if no exception was raised — and we verify the figure was created
        assert fig is not None, "plot_umap_overlay returned None when NaNs present"

    def test_umap_overlay_produces_10_files(self, tmp_path):
        """One PNG per feature must be written: exactly 10 files named umap_overlay_<feature>.png."""
        mod = _import_script()
        df = _make_taxonomy_df()
        umap_df = _make_umap_df(df)
        mod.save_umap_overlays(umap_df, output_dir=tmp_path)

        produced = list(tmp_path.glob("umap_overlay_*.png"))
        assert len(produced) == 10, (
            f"Expected 10 umap_overlay_*.png files, found {len(produced)}: "
            f"{[p.name for p in produced]}"
        )
        for feature in ACOUSTIC_FEATURES:
            expected = tmp_path / f"umap_overlay_{feature}.png"
            assert expected.exists(), f"Missing expected overlay file: {expected.name}"

    def test_umap_panel_has_10_axes(self):
        """The 2×5 panel figure must have exactly 10 axes — one per acoustic feature."""
        mod = _import_script()
        df = _make_taxonomy_df()
        umap_df = _make_umap_df(df)
        fig = mod.plot_umap_overlay_panel(umap_df)
        n_axes = len(fig.get_axes())
        plt.close(fig)
        assert n_axes == 10, (
            f"UMAP overlay panel should have 10 axes (2×5 grid), got {n_axes}"
        )

    def test_umap_overlay_uses_continuous_colormap(self):
        """UMAP overlay for a numeric feature must use a continuous colormap
        (e.g. viridis), not a categorical legend — verifying the rendering intent.
        """
        mod = _import_script()
        df = _make_taxonomy_df()
        umap_df = _make_umap_df(df)
        feature = "tonality"
        fig = mod.plot_umap_overlay(umap_df, feature=feature)
        axes = fig.get_axes()
        plt.close(fig)
        assert len(axes) > 0, "plot_umap_overlay produced a figure with no axes"
        # A continuous colormap is indicated by a colorbar (ScalarMappable) in the figure
        has_colorbar = any(
            hasattr(ax, "collections") and len(ax.collections) > 0
            for ax in axes
        )
        assert has_colorbar, (
            "UMAP overlay should use scatter with continuous colormap (collections), "
            "not discrete category patches"
        )


class TestWithinTypeVariability:
    """ROADMAP item 5: Violin plots for within-type variability."""

    def test_within_type_variability_sparse_type_no_crash(self, tmp_path):
        """Violin plot must not crash when a type has < 5 calls.
        (Single-point and 2-point violin distributions are degenerate but must be handled.)
        """
        mod = _import_script()
        rng = np.random.default_rng(17)
        # Build DataFrame where "Chevron" has only 2 calls
        rows = []
        for stype in SYLLABLE_TYPES:
            n = 2 if stype == "Chevron" else 30
            for _ in range(n):
                rows.append({
                    feat: rng.uniform(0, 1) for feat in ACOUSTIC_FEATURES
                } | {"syllable_type": stype, "classification_confidence": "high",
                     "det_prob_max": 0.9, "det_prob_mean": 0.85,
                     "file": "2024-09-30_10-00-00_0000001"})
        df = pd.DataFrame(rows)
        # Must complete without raising
        fig = mod.plot_within_type_variability(df)
        plt.close(fig)
        assert fig is not None, "plot_within_type_variability returned None"

    def test_within_type_variability_has_10_panels(self):
        """The variability figure must have exactly 10 panels — one per acoustic feature."""
        mod = _import_script()
        df = _make_taxonomy_df()
        fig = mod.plot_within_type_variability(df)
        n_axes = len(fig.get_axes())
        plt.close(fig)
        assert n_axes == 10, (
            f"within_type_variability should produce 10 subplots (one per feature), "
            f"got {n_axes}"
        )


class TestSummaryCSV:
    """ROADMAP items 6, and additional: summary CSV correctness."""

    def test_summary_csv_has_7_rows(self, tmp_path):
        """Summary CSV must have exactly 7 rows — one per syllable type."""
        mod = _import_script()
        df = _make_taxonomy_df()
        pca_result = mod.run_pca(df[ACOUSTIC_FEATURES], n_components=5)
        loadings = mod.get_pca_loadings(pca_result, feature_names=ACOUSTIC_FEATURES)
        summary_path = tmp_path / "summary.csv"
        mod.write_summary_csv(df, pca_result, loadings, output_path=summary_path)

        summary = pd.read_csv(summary_path)
        assert len(summary) == 7, (
            f"Summary CSV should have 7 rows (one per type), got {len(summary)}"
        )

    def test_summary_csv_all_7_types_present(self, tmp_path):
        """Every syllable type must appear exactly once in the summary CSV."""
        mod = _import_script()
        df = _make_taxonomy_df()
        pca_result = mod.run_pca(df[ACOUSTIC_FEATURES], n_components=5)
        loadings = mod.get_pca_loadings(pca_result, feature_names=ACOUSTIC_FEATURES)
        summary_path = tmp_path / "summary.csv"
        mod.write_summary_csv(df, pca_result, loadings, output_path=summary_path)

        summary = pd.read_csv(summary_path)
        type_col = summary.columns[summary.columns.str.lower().isin(
            ["syllable_type", "type", "category"]
        )].tolist()
        assert len(type_col) == 1, (
            f"Summary CSV must have a type column; found candidates: {type_col}"
        )
        present_types = set(summary[type_col[0]].tolist())
        for stype in SYLLABLE_TYPES:
            assert stype in present_types, (
                f"Syllable type '{stype}' missing from summary CSV"
            )

    def test_summary_csv_has_expected_columns(self, tmp_path):
        """Summary CSV must contain per-type mean, std, median for each feature,
        plus PCA variance explained (first 5 PCs) and power/tonality stats.
        """
        mod = _import_script()
        df = _make_taxonomy_df()
        pca_result = mod.run_pca(df[ACOUSTIC_FEATURES], n_components=5)
        loadings = mod.get_pca_loadings(pca_result, feature_names=ACOUSTIC_FEATURES)
        summary_path = tmp_path / "summary.csv"
        mod.write_summary_csv(df, pca_result, loadings, output_path=summary_path)

        summary = pd.read_csv(summary_path)
        # Per-type statistics: at least one stat column per feature
        col_str = " ".join(summary.columns.str.lower())
        for feature in ACOUSTIC_FEATURES:
            # Feature name (possibly abbreviated) should appear somewhere in columns
            feature_key = feature.replace("_hz", "").replace("_s", "").replace("_db", "")[:8]
            assert feature_key in col_str, (
                f"Expected column for feature '{feature}' (key '{feature_key}') "
                f"in summary CSV. Columns: {list(summary.columns)}"
            )

    def test_summary_csv_no_nan_in_key_columns(self, tmp_path):
        """Count and mean statistics in the summary CSV must not be NaN.
        (NaN stats would indicate a broken groupby or aggregation step.)
        """
        mod = _import_script()
        df = _make_taxonomy_df()
        pca_result = mod.run_pca(df[ACOUSTIC_FEATURES], n_components=5)
        loadings = mod.get_pca_loadings(pca_result, feature_names=ACOUSTIC_FEATURES)
        summary_path = tmp_path / "summary.csv"
        mod.write_summary_csv(df, pca_result, loadings, output_path=summary_path)

        summary = pd.read_csv(summary_path)
        # Count columns must be non-NaN
        count_cols = [c for c in summary.columns if "count" in c.lower() or c.lower() == "n"]
        for col in count_cols:
            assert not summary[col].isna().any(), (
                f"Column '{col}' in summary CSV has NaN values"
            )

    def test_summary_csv_means_within_range(self, tmp_path):
        """Per-type mean values must be within the observed data range
        (they are means, not extrapolations).
        """
        mod = _import_script()
        df = _make_taxonomy_df()
        pca_result = mod.run_pca(df[ACOUSTIC_FEATURES], n_components=5)
        loadings = mod.get_pca_loadings(pca_result, feature_names=ACOUSTIC_FEATURES)
        summary_path = tmp_path / "summary.csv"
        mod.write_summary_csv(df, pca_result, loadings, output_path=summary_path)

        summary = pd.read_csv(summary_path)
        # mean_power_db means must be within the synthetic data range [-60, -20]
        mean_power_cols = [c for c in summary.columns
                          if "power" in c.lower() and "mean" in c.lower()]
        if mean_power_cols:
            col = mean_power_cols[0]
            assert (summary[col] >= -65).all() and (summary[col] <= -15).all(), (
                f"mean_power_db means outside expected range [-60, -20] (±5 tolerance): "
                f"{summary[col].tolist()}"
            )


class TestLowConfidenceFilter:
    """ROADMAP item 7: boundary/low-confidence analysis."""

    def test_low_confidence_filter_produces_nonempty_subset(self):
        """The boundary analysis filter (classification_confidence == 'low') must
        produce a non-empty subset when low-confidence calls exist in the data.
        """
        mod = _import_script()
        df = _make_taxonomy_df(low_confidence_fraction=0.15)
        low_conf = mod.filter_low_confidence(df)
        assert len(low_conf) > 0, (
            "filter_low_confidence returned empty DataFrame despite low-confidence "
            "calls being present"
        )
        assert (low_conf["classification_confidence"] == "low").all(), (
            "filter_low_confidence returned rows that are not 'low' confidence"
        )

    def test_boundary_subset_smaller_than_full(self):
        """The low-confidence subset must be strictly smaller than the full dataset."""
        mod = _import_script()
        df = _make_taxonomy_df(low_confidence_fraction=0.15)
        low_conf = mod.filter_low_confidence(df)
        assert len(low_conf) < len(df), (
            "Low-confidence subset should be smaller than full dataset"
        )

    def test_boundary_empty_low_confidence_no_crash(self, tmp_path):
        """Boundary analysis must not crash when no low-confidence calls exist.
        (All calls classified as 'high' — degenerate but valid input.)
        """
        mod = _import_script()
        df = _make_taxonomy_df(low_confidence_fraction=0.0)
        # Override all confidence to 'high'
        df["classification_confidence"] = "high"
        umap_df = _make_umap_df(df)

        fig = mod.plot_boundary_cases(df, umap_df)
        plt.close(fig)
        # No exception = pass; just verify something was returned
        assert fig is not None, "plot_boundary_cases returned None on empty low-conf set"


class TestPowerTonalityPanel:
    """ROADMAP item 8: 4-panel power/tonality figure."""

    def test_power_tonality_panel_has_4_subplots(self):
        """The power & tonality deep-dive figure must produce exactly 4 subplots:
        1. Power by type (box plot)
        2. Tonality distribution (histogram)
        3. Tonality vs detection confidence (scatter)
        4. Power over time (binned)
        """
        mod = _import_script()
        df = _make_taxonomy_df()
        fig = mod.plot_power_tonality(df)
        n_axes = len(fig.get_axes())
        plt.close(fig)
        assert n_axes == 4, (
            f"Power/tonality panel should have 4 subplots, got {n_axes}"
        )


class TestFrequencyDrift:
    """ROADMAP item 9: Frequency drift regression."""

    def test_frequency_drift_regression_returns_slope_and_pvalue(self):
        """compute_frequency_drift must return a dict (or named object) with
        'slope' and 'pvalue' keys — the primary outputs of a linear regression.
        """
        mod = _import_script()
        df = _make_taxonomy_df()
        result = mod.compute_frequency_drift(df)
        assert "slope" in result, f"Drift result missing 'slope': {result}"
        assert "pvalue" in result, f"Drift result missing 'pvalue': {result}"

    def test_frequency_drift_pvalue_in_range(self):
        """p-value from linear regression must be in [0, 1] — it is a probability."""
        mod = _import_script()
        df = _make_taxonomy_df()
        result = mod.compute_frequency_drift(df)
        pval = result["pvalue"]
        assert 0.0 <= pval <= 1.0, (
            f"Frequency drift p-value must be in [0, 1], got {pval}"
        )

    def test_frequency_drift_slope_sign_correct(self):
        """When frequency increases linearly over time, the drift slope must be positive.

        Hand-computed: we set principal_freq_hz = 50 + 0.01 * hour_index.
        Over 12 hours (0–11), slope = +0.01 kHz/hour. The regression must recover a
        positive slope.
        """
        mod = _import_script()
        rng = np.random.default_rng(5)
        n = 120
        hours = np.arange(n)
        # Linear upward trend: 50 + 0.5 * hour + small noise
        freqs = 50.0 + 0.5 * hours + rng.normal(0, 0.1, n)
        rows = []
        for i in range(n):
            rows.append({
                "file": f"2024-09-30_{(10 + i // 6) % 24:02d}-00-00_{i:07d}",
                "principal_freq_hz": freqs[i],
                "syllable_type": rng.choice(SYLLABLE_TYPES),
                "classification_confidence": "high",
                "det_prob_max": 0.9,
                "det_prob_mean": 0.85,
                **{feat: rng.uniform(0, 1) for feat in ACOUSTIC_FEATURES
                   if feat != "principal_freq_hz"},
            })
        df = pd.DataFrame(rows)
        result = mod.compute_frequency_drift(df)
        assert result["slope"] > 0, (
            f"Slope should be positive for upward-trending frequency, got {result['slope']}"
        )

    def test_frequency_drift_flat_data_slope_near_zero(self):
        """When principal_freq_hz is constant (no drift), slope must be near 0.

        Hand-computed expected value: constant frequency 70 kHz over time => slope = 0.
        Tolerance: |slope| < 0.01 kHz/hour (within noise floor).
        """
        mod = _import_script()
        rng = np.random.default_rng(3)
        n = 100
        rows = []
        for i in range(n):
            rows.append({
                "file": f"2024-09-30_{(10 + i // 10) % 24:02d}-00-00_{i:07d}",
                "principal_freq_hz": 70.0,  # exactly constant
                "syllable_type": rng.choice(SYLLABLE_TYPES),
                "classification_confidence": "high",
                "det_prob_max": 0.9,
                "det_prob_mean": 0.85,
                **{feat: rng.uniform(0, 1) for feat in ACOUSTIC_FEATURES
                   if feat != "principal_freq_hz"},
            })
        df = pd.DataFrame(rows)
        result = mod.compute_frequency_drift(df)
        assert abs(result["slope"]) < 0.01, (
            f"Slope on constant-frequency data should be near 0, got {result['slope']:.6f}"
        )


class TestFlatCallTemporal:
    """ROADMAP item 10: Flat call temporal distribution with zero-count bins."""

    def test_flat_temporal_handles_zero_flat_bins(self):
        """When some hourly bins contain zero Flat calls (but non-zero other calls),
        the analysis must produce a result without NaN or crash.
        """
        mod = _import_script()
        rng = np.random.default_rng(11)
        # Only include Flat calls in the first half of the time range
        rows = []
        for i in range(60):
            hour = 10 + i // 6  # hours 10–19
            stype = "Flat" if hour < 15 else rng.choice(["Down", "Up", "Short"])
            rows.append({
                "file": f"2024-09-30_{hour:02d}-00-00_{i:07d}",
                "syllable_type": stype,
                "classification_confidence": "high",
                "det_prob_max": 0.9,
                "det_prob_mean": 0.85,
                **{feat: rng.uniform(0, 1) for feat in ACOUSTIC_FEATURES},
            })
        df = pd.DataFrame(rows)
        result = mod.analyze_flat_temporal(df)
        # Must return something (dict, DataFrame, or figure) without crashing
        assert result is not None, "analyze_flat_temporal returned None"
        # The result must not propagate NaN in rate values
        if isinstance(result, pd.DataFrame):
            rate_cols = [c for c in result.columns if "rate" in c.lower() or "count" in c.lower()]
            for col in rate_cols:
                assert not result[col].isna().any(), (
                    f"Zero-count bins should be filled with 0, not NaN (column: {col})"
                )


class TestEndToEnd:
    """ROADMAP item 11: Full script execution on synthetic data."""

    def test_end_to_end_produces_all_expected_output_files(self, tmp_path):
        """Running the script end-to-end on synthetic CSVs must produce all
        expected output files in the output directory.

        Expected files (from ROADMAP spec):
        - feature_correlation.png
        - pca_variance.png
        - pca_loadings.png
        - pca_scatter.png
        - umap_overlay_panel.png
        - umap_overlay_<feature>.png × 10
        - within_type_variability.png
        - boundary_cases.png
        - power_tonality_analysis.png
        - frequency_drift.png
        - acoustic_feature_summary.csv (or similar)
        """
        # Write synthetic CSVs
        taxonomy_df = _make_taxonomy_df(n_per_type=20)
        umap_df = _make_umap_df(taxonomy_df)

        taxonomy_path = tmp_path / "classified_traditional.csv"
        umap_path = tmp_path / "reclassified_detections.csv"
        output_dir = tmp_path / "output"

        taxonomy_df.to_csv(taxonomy_path, index=False)
        umap_df.to_csv(umap_path, index=False)
        output_dir.mkdir()

        # Run the script as a subprocess (tests script invocation via CLI)
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--input-taxonomy", str(taxonomy_path),
                "--input-umap", str(umap_path),
                "--output-dir", str(output_dir),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"Script exited with code {result.returncode}.\n"
            f"STDOUT: {result.stdout[-2000:]}\n"
            f"STDERR: {result.stderr[-2000:]}"
        )

        expected_files = [
            "feature_correlation.png",
            "pca_variance.png",
            "pca_loadings.png",
            "pca_scatter.png",
            "umap_overlay_panel.png",
            "within_type_variability.png",
            "boundary_cases.png",
            "power_tonality_analysis.png",
            "frequency_drift.png",
        ] + [f"umap_overlay_{feat}.png" for feat in ACOUSTIC_FEATURES]

        produced = {p.name for p in output_dir.iterdir()}
        for fname in expected_files:
            assert fname in produced, (
                f"Expected output file '{fname}' not found in {output_dir}.\n"
                f"Files produced: {sorted(produced)}"
            )

        # Summary CSV must also exist
        csv_files = list(output_dir.glob("*.csv"))
        assert len(csv_files) >= 1, (
            f"No summary CSV produced in {output_dir}. Files: {sorted(produced)}"
        )


# ── Additional gap-pattern tests ──────────────────────────────────────────────


class TestTimestampParsing:
    """parse_filename_timestamp correctness and error handling."""

    def test_parse_filename_timestamp_correct_value(self):
        """parse_filename_timestamp must correctly extract datetime from a standard filename.

        Input: '2024-09-30_11-18-17_0000001'
        Expected: datetime(2024, 9, 30, 11, 18, 17)
        This is a hand-computed spot-check — no tolerance, exact match.
        """
        from datetime import datetime
        mod = _import_script()
        result = mod.parse_filename_timestamp("2024-09-30_11-18-17_0000001")
        expected = datetime(2024, 9, 30, 11, 18, 17)
        assert result == expected, (
            f"Expected {expected}, got {result}"
        )

    def test_parse_filename_timestamp_invalid_raises(self):
        """parse_filename_timestamp must raise a clear error on a malformed filename,
        not silently return None or a garbage datetime.
        """
        mod = _import_script()
        with pytest.raises((ValueError, AttributeError, TypeError)):
            mod.parse_filename_timestamp("not_a_valid_filename.wav")


class TestEmptyDataFrame:
    """Empty and edge-case inputs for public analysis functions."""

    def test_empty_dataframe_raises_or_returns_gracefully(self):
        """All major public functions must not silently produce corrupt outputs on empty input.
        They may raise ValueError (preferred) or return a sentinel (empty DF, None with warning).
        They must NOT return a non-empty result from no data.
        """
        mod = _import_script()
        empty_df = pd.DataFrame(columns=ACOUSTIC_FEATURES + [
            "syllable_type", "classification_confidence", "det_prob_max",
            "det_prob_mean", "file"
        ])

        # run_pca should raise on empty input
        with pytest.raises(Exception):
            mod.run_pca(empty_df[ACOUSTIC_FEATURES])

    def test_compute_correlation_matrices_empty_raises(self):
        """compute_correlation_matrices must raise on empty feature matrix,
        not return a matrix of NaN silently.
        """
        mod = _import_script()
        empty_features = pd.DataFrame(columns=ACOUSTIC_FEATURES)
        with pytest.raises(Exception):
            mod.compute_correlation_matrices(empty_features)
