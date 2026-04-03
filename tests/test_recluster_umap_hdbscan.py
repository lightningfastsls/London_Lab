"""Tests for scripts/recluster_umap_hdbscan.py — written by test-architect BEFORE implementation.

ROADMAP test plan coverage (from docs/handoffs/umap-hdbscan-recluster.md):
  1. load_and_prepare: NaN handling              -> test_load_and_prepare_drops_nan_rows
  2. load_and_prepare: feature exclusion         -> test_load_and_prepare_feature_exclusion
  3. load_and_prepare: scaling output shape      -> test_load_and_prepare_scaling_shape_and_stats
  4. build_output_df: row count preservation     -> test_build_output_df_preserves_row_count
  5. build_output_df: NaN row labeling           -> test_build_output_df_nan_rows_get_noise_label
  6. build_output_df: new columns added          -> test_build_output_df_adds_four_columns
  7. compute_cluster_summary: aggregation        -> test_compute_cluster_summary_correct_aggregation
  8. compute_cluster_summary: count column       -> test_compute_cluster_summary_has_count_column
  9. plot_umap_scatter: file created             -> test_plot_umap_scatter_creates_png
  10. plot_contingency_matrix: file created      -> test_plot_contingency_matrix_creates_png
  11. run_umap: output shape                     -> test_run_umap_output_shape
  12. run_hdbscan: output shape and types        -> test_run_hdbscan_output_shapes_and_types
  13. Edge case: all noise                       -> test_run_hdbscan_all_noise_case
  14. Edge case: single feature                  -> test_load_and_prepare_single_feature_after_exclusion
  15. Integration: small dataset full pipeline   -> test_integration_small_dataset_full_pipeline

Additional coverage (recurring gap patterns):
  - Empty exclude list does not drop features   -> test_load_and_prepare_no_exclusions_uses_all_features
  - Z-score: known input produces zero-mean     -> test_load_and_prepare_scaled_features_zero_mean
  - Z-score: known input produces unit-std      -> test_load_and_prepare_scaled_features_unit_std
  - build_output_df: index is sorted            -> test_build_output_df_index_is_sorted
  - build_output_df: NaN rows have NaN umap_x   -> test_build_output_df_nan_rows_have_nan_umap_coords
  - build_output_df: prob 0.0 for NaN rows      -> test_build_output_df_nan_rows_prob_zero
  - compute_cluster_summary: noise group present-> test_compute_cluster_summary_includes_noise_group
  - run_umap: 2D and 8D shapes both work        -> test_run_umap_2d_and_8d_shapes
  - HDBSCAN probabilities in [0, 1]             -> test_run_hdbscan_probabilities_in_range
  - load_and_prepare: returns df_full length    -> test_load_and_prepare_df_full_preserves_all_rows
  - CSV missing column raises immediately       -> test_load_and_prepare_missing_feature_column_raises
  - Excluded feature not in X_scaled            -> test_load_and_prepare_excluded_feature_absent_from_X
  - build_output_df: original columns preserved -> test_build_output_df_preserves_original_columns

Total: 28 tests (15 from ROADMAP, 13 additional)
"""

from __future__ import annotations

import sys
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path bootstrap — script lives in scripts/ (one level below repo root)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Deferred import — tests fail individually if the module is missing rather
# than the whole file failing at collection time.
# ---------------------------------------------------------------------------

def _import_module():
    """Return the recluster_umap_hdbscan module or raise ImportError."""
    import recluster_umap_hdbscan as m  # noqa: PLC0415
    return m


# ---------------------------------------------------------------------------
# Helpers for building synthetic DataFrames
# ---------------------------------------------------------------------------

ACOUSTIC_FEATURES = [
    "call_length_s",
    "principal_freq_hz",
    "low_freq_hz",
    "high_freq_hz",
    "bandwidth_hz",
    "freq_std_dev_hz",
    "slope",
    "sinuosity",
    "tonality",
    "mean_power_db",
]

_RNG = np.random.default_rng(0)


def _make_synthetic_df(
    n_valid: int = 50,
    n_nan: int = 5,
    extra_cols: dict | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic DataFrame matching the expected CSV structure.

    Parameters
    ----------
    n_valid:
        Number of rows with complete acoustic feature data.
    n_nan:
        Number of rows where all acoustic features are NaN.
    extra_cols:
        Additional columns to add (e.g. {"label": ...}).
    seed:
        NumPy random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    n_total = n_valid + n_nan

    data: dict = {}

    # Acoustic features — typical USV value ranges from the spec
    data["call_length_s"] = np.where(
        np.arange(n_total) < n_valid,
        rng.uniform(0.01, 0.5, n_total),
        np.nan,
    )
    data["principal_freq_hz"] = np.where(
        np.arange(n_total) < n_valid,
        rng.uniform(47, 91, n_total),
        np.nan,
    )
    data["low_freq_hz"] = np.where(
        np.arange(n_total) < n_valid,
        rng.uniform(40, 70, n_total),
        np.nan,
    )
    data["high_freq_hz"] = np.where(
        np.arange(n_total) < n_valid,
        rng.uniform(70, 110, n_total),
        np.nan,
    )
    data["bandwidth_hz"] = np.where(
        np.arange(n_total) < n_valid,
        rng.uniform(1, 50, n_total),
        np.nan,
    )
    data["freq_std_dev_hz"] = np.where(
        np.arange(n_total) < n_valid,
        rng.uniform(1, 30, n_total),
        np.nan,
    )
    data["slope"] = np.where(
        np.arange(n_total) < n_valid,
        rng.uniform(-793, 704, n_total),
        np.nan,
    )
    data["sinuosity"] = np.where(
        np.arange(n_total) < n_valid,
        rng.uniform(1.2, 8.1, n_total),
        np.nan,
    )
    data["tonality"] = np.where(
        np.arange(n_total) < n_valid,
        rng.uniform(0.18, 0.34, n_total),
        np.nan,
    )
    data["mean_power_db"] = np.where(
        np.arange(n_total) < n_valid,
        rng.uniform(-80, -20, n_total),
        np.nan,
    )

    # Non-feature columns that should be preserved
    data["label"] = [f"Cluster_{rng.integers(1, 28)}" for _ in range(n_total)]
    data["wav_stem"] = [f"recording_{i:04d}" for i in range(n_total)]

    if extra_cols:
        data.update(extra_cols)

    return pd.DataFrame(data)


def _write_synthetic_csv(
    path: Path,
    n_valid: int = 50,
    n_nan: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Write a synthetic CSV to *path* and return the DataFrame."""
    df = _make_synthetic_df(n_valid=n_valid, n_nan=n_nan, seed=seed)
    df.to_csv(path, index=False)
    return df


# ===========================================================================
# load_and_prepare tests
# ===========================================================================

class TestLoadAndPrepare:
    """Tests for load_and_prepare(csv_path, exclude_features)."""

    def test_load_and_prepare_drops_nan_rows(self, tmp_path):
        """Rows with NaN in any acoustic feature must not appear in df_valid.

        Spec: 'Drops rows where ANY feature is NaN' (from handoff).
        """
        m = _import_module()
        csv = tmp_path / "test.csv"
        df_orig = _write_synthetic_csv(csv, n_valid=40, n_nan=7)

        df_full, df_valid, X_scaled, features = m.load_and_prepare(
            str(csv), exclude_features=[]
        )

        assert len(df_valid) == 40, (
            f"Expected 40 valid rows, got {len(df_valid)}. "
            "NaN rows must be excluded from df_valid."
        )

    def test_load_and_prepare_df_full_preserves_all_rows(self, tmp_path):
        """df_full must contain ALL rows including NaN rows.

        Spec: 'kept in output with label -1'.
        """
        m = _import_module()
        csv = tmp_path / "test.csv"
        _write_synthetic_csv(csv, n_valid=40, n_nan=7)

        df_full, df_valid, X_scaled, features = m.load_and_prepare(
            str(csv), exclude_features=[]
        )

        assert len(df_full) == 47, (
            f"df_full must have all {47} rows, got {len(df_full)}."
        )

    def test_load_and_prepare_scaling_shape_and_stats(self, tmp_path):
        """X_scaled must have shape (n_valid, n_features) with all 10 features."""
        m = _import_module()
        csv = tmp_path / "test.csv"
        _write_synthetic_csv(csv, n_valid=30, n_nan=3)

        _, _, X_scaled, features = m.load_and_prepare(
            str(csv), exclude_features=[]
        )

        assert X_scaled.shape == (30, 10), (
            f"Expected (30, 10), got {X_scaled.shape}. "
            "Shape must match (n_valid, n_features)."
        )

    def test_load_and_prepare_scaled_features_zero_mean(self, tmp_path):
        """StandardScaler must produce zero mean across each feature column.

        Hand-computed invariant: sum of scaled values / N == 0 to floating-point precision.
        """
        m = _import_module()
        csv = tmp_path / "test.csv"
        _write_synthetic_csv(csv, n_valid=100, n_nan=0)

        _, _, X_scaled, _ = m.load_and_prepare(str(csv), exclude_features=[])

        col_means = X_scaled.mean(axis=0)
        np.testing.assert_allclose(
            col_means, 0.0, atol=1e-10,
            err_msg="StandardScaler must produce zero mean for each feature column.",
        )

    def test_load_and_prepare_scaled_features_unit_std(self, tmp_path):
        """StandardScaler must produce unit standard deviation per feature.

        Without this the slope column (range ~1500) would dominate UMAP — the spec
        explicitly flags this risk.
        """
        m = _import_module()
        csv = tmp_path / "test.csv"
        _write_synthetic_csv(csv, n_valid=100, n_nan=0)

        _, _, X_scaled, _ = m.load_and_prepare(str(csv), exclude_features=[])

        col_stds = X_scaled.std(axis=0, ddof=0)
        np.testing.assert_allclose(
            col_stds, 1.0, atol=1e-10,
            err_msg="StandardScaler must produce unit std for each feature column.",
        )

    def test_load_and_prepare_feature_exclusion(self, tmp_path):
        """--exclude-features must remove named features from X_scaled and feature_names."""
        m = _import_module()
        csv = tmp_path / "test.csv"
        _write_synthetic_csv(csv, n_valid=30, n_nan=0)

        _, _, X_scaled, features = m.load_and_prepare(
            str(csv), exclude_features=["mean_power_db", "slope"]
        )

        assert "mean_power_db" not in features, "mean_power_db must be excluded."
        assert "slope" not in features, "slope must be excluded."
        assert X_scaled.shape[1] == 8, (
            f"Expected 8 columns after excluding 2, got {X_scaled.shape[1]}."
        )

    def test_load_and_prepare_no_exclusions_uses_all_features(self, tmp_path):
        """Empty exclude list must keep all 10 ACOUSTIC_FEATURES."""
        m = _import_module()
        csv = tmp_path / "test.csv"
        _write_synthetic_csv(csv, n_valid=20, n_nan=0)

        _, _, X_scaled, features = m.load_and_prepare(
            str(csv), exclude_features=[]
        )

        assert len(features) == 10, f"Expected 10 features, got {len(features)}."
        assert set(features) == set(ACOUSTIC_FEATURES)

    def test_load_and_prepare_single_feature_after_exclusion(self, tmp_path):
        """Excluding 9 of 10 features must still produce valid (N, 1) X_scaled.

        Edge case: single remaining feature must still be normalized.
        """
        m = _import_module()
        csv = tmp_path / "test.csv"
        _write_synthetic_csv(csv, n_valid=20, n_nan=0)

        exclude_nine = [f for f in ACOUSTIC_FEATURES if f != "call_length_s"]
        _, _, X_scaled, features = m.load_and_prepare(
            str(csv), exclude_features=exclude_nine
        )

        assert X_scaled.shape[1] == 1, f"Expected (20, 1), got {X_scaled.shape}."
        assert features == ["call_length_s"]

    def test_load_and_prepare_excluded_feature_absent_from_X(self, tmp_path):
        """Values from excluded feature must not appear in X_scaled columns."""
        m = _import_module()
        csv = tmp_path / "test.csv"
        df = _make_synthetic_df(n_valid=30, n_nan=0)
        # Set mean_power_db to a recognizable constant so we can detect its presence
        df["mean_power_db"] = 9999.0
        df.to_csv(csv, index=False)

        _, df_valid, X_scaled, features = m.load_and_prepare(
            str(csv), exclude_features=["mean_power_db"]
        )

        # If mean_power_db leaked in, some column would be all-9999 before scaling
        assert "mean_power_db" not in features
        # After normalization the constant 9999 column would have std=0 → NaN or inf
        assert np.all(np.isfinite(X_scaled)), (
            "Excluded constant column leaked into X_scaled."
        )

    def test_load_and_prepare_missing_feature_column_raises(self, tmp_path):
        """CSV missing an expected acoustic feature column must raise an error.

        This ensures the caller gets a clear failure rather than silent wrong output.
        """
        m = _import_module()
        csv = tmp_path / "bad.csv"
        df = _make_synthetic_df(n_valid=10, n_nan=0)
        df = df.drop(columns=["call_length_s"])
        df.to_csv(csv, index=False)

        with pytest.raises((KeyError, ValueError)):
            m.load_and_prepare(str(csv), exclude_features=[])


# ===========================================================================
# build_output_df tests
# ===========================================================================

class TestBuildOutputDf:
    """Tests for build_output_df(df_full, df_valid, emb_2d, labels, probabilities)."""

    def _make_inputs(self, n_valid=40, n_nan=5, seed=42):
        """Build consistent synthetic inputs for build_output_df."""
        df_full = _make_synthetic_df(n_valid=n_valid, n_nan=n_nan, seed=seed)
        df_valid = df_full.iloc[:n_valid].copy()
        rng = np.random.default_rng(seed)
        emb_2d = rng.standard_normal((n_valid, 2)).astype(np.float32)
        labels = rng.integers(0, 5, n_valid)
        probabilities = rng.uniform(0, 1, n_valid)
        return df_full, df_valid, emb_2d, labels, probabilities

    def test_build_output_df_preserves_row_count(self):
        """Output must have exactly the same number of rows as df_full."""
        m = _import_module()
        df_full, df_valid, emb_2d, labels, probs = self._make_inputs(40, 5)

        df_out = m.build_output_df(df_full, df_valid, emb_2d, labels, probs)

        assert len(df_out) == 45, (
            f"Expected 45 rows (40 valid + 5 NaN), got {len(df_out)}."
        )

    def test_build_output_df_adds_four_columns(self):
        """Output must add exactly these four new columns to the DataFrame."""
        m = _import_module()
        df_full, df_valid, emb_2d, labels, probs = self._make_inputs(30, 3)
        original_cols = set(df_full.columns)

        df_out = m.build_output_df(df_full, df_valid, emb_2d, labels, probs)

        new_cols = set(df_out.columns) - original_cols
        assert new_cols == {"umap_x", "umap_y", "hdbscan_label", "hdbscan_probability"}, (
            f"Expected exactly four new columns, got: {new_cols}"
        )

    def test_build_output_df_nan_rows_get_noise_label(self):
        """NaN rows must receive hdbscan_label == -1."""
        m = _import_module()
        df_full, df_valid, emb_2d, labels, probs = self._make_inputs(40, 5)

        df_out = m.build_output_df(df_full, df_valid, emb_2d, labels, probs)

        # The last 5 rows in df_full were the NaN rows
        nan_indices = df_full.index.difference(df_valid.index)
        nan_labels = df_out.loc[nan_indices, "hdbscan_label"].values
        assert np.all(nan_labels == -1), (
            f"NaN rows must have hdbscan_label=-1, got: {nan_labels}"
        )

    def test_build_output_df_nan_rows_prob_zero(self):
        """NaN rows must receive hdbscan_probability == 0.0."""
        m = _import_module()
        df_full, df_valid, emb_2d, labels, probs = self._make_inputs(40, 5)

        df_out = m.build_output_df(df_full, df_valid, emb_2d, labels, probs)

        nan_indices = df_full.index.difference(df_valid.index)
        nan_probs = df_out.loc[nan_indices, "hdbscan_probability"].values
        np.testing.assert_array_equal(
            nan_probs, 0.0,
            err_msg="NaN rows must have hdbscan_probability=0.0.",
        )

    def test_build_output_df_nan_rows_have_nan_umap_coords(self):
        """NaN rows must have NaN for umap_x and umap_y (they were never embedded)."""
        m = _import_module()
        df_full, df_valid, emb_2d, labels, probs = self._make_inputs(40, 5)

        df_out = m.build_output_df(df_full, df_valid, emb_2d, labels, probs)

        nan_indices = df_full.index.difference(df_valid.index)
        assert df_out.loc[nan_indices, "umap_x"].isna().all(), (
            "NaN rows must have NaN umap_x coordinates."
        )
        assert df_out.loc[nan_indices, "umap_y"].isna().all(), (
            "NaN rows must have NaN umap_y coordinates."
        )

    def test_build_output_df_index_is_sorted(self):
        """Output index must be monotonically increasing (preserves original row order)."""
        m = _import_module()
        df_full, df_valid, emb_2d, labels, probs = self._make_inputs(40, 5)

        df_out = m.build_output_df(df_full, df_valid, emb_2d, labels, probs)

        idx = df_out.index.tolist()
        assert idx == sorted(idx), "Output index must be sorted (original row order preserved)."

    def test_build_output_df_preserves_original_columns(self):
        """All original columns from df_full must be present in df_out."""
        m = _import_module()
        df_full, df_valid, emb_2d, labels, probs = self._make_inputs(30, 3)
        original_cols = set(df_full.columns)

        df_out = m.build_output_df(df_full, df_valid, emb_2d, labels, probs)

        missing = original_cols - set(df_out.columns)
        assert not missing, f"Original columns missing from output: {missing}"

    def test_build_output_df_valid_rows_get_real_embedding(self):
        """Valid rows must carry the UMAP coordinates from emb_2d, not NaN."""
        m = _import_module()
        df_full, df_valid, emb_2d, labels, probs = self._make_inputs(20, 0)

        df_out = m.build_output_df(df_full, df_valid, emb_2d, labels, probs)

        assert df_out["umap_x"].notna().all(), (
            "When there are no NaN rows, umap_x must be fully non-NaN."
        )
        assert df_out["umap_y"].notna().all()


# ===========================================================================
# compute_cluster_summary tests
# ===========================================================================

class TestComputeClusterSummary:
    """Tests for compute_cluster_summary(df_valid, features, label_col)."""

    def _make_labeled_df(self, seed=0):
        """Return a small df_valid with hdbscan_label column."""
        rng = np.random.default_rng(seed)
        n = 60
        features = ["call_length_s", "slope", "tonality"]
        data = {f: rng.standard_normal(n) for f in features}
        data["hdbscan_label"] = np.array([0] * 20 + [1] * 20 + [-1] * 20)
        return pd.DataFrame(data), features

    def test_compute_cluster_summary_has_count_column(self):
        """Output DataFrame must have a 'count' column."""
        m = _import_module()
        df, features = self._make_labeled_df()

        summary = m.compute_cluster_summary(df, features, label_col="hdbscan_label")

        assert "count" in summary.columns, (
            "compute_cluster_summary must produce a 'count' column."
        )

    def test_compute_cluster_summary_correct_aggregation(self):
        """Count column must equal actual group sizes.

        Hand-computed expected values: 20 per label group.
        """
        m = _import_module()
        df, features = self._make_labeled_df()

        summary = m.compute_cluster_summary(df, features, label_col="hdbscan_label")

        counts = summary.set_index(summary.columns[0])["count"].to_dict()
        for lbl, expected_count in {0: 20, 1: 20, -1: 20}.items():
            assert lbl in counts, f"Label {lbl} missing from summary."
            assert counts[lbl] == expected_count, (
                f"Label {lbl}: expected count=20, got {counts[lbl]}."
            )

    def test_compute_cluster_summary_includes_noise_group(self):
        """HDBSCAN noise (label -1) must appear as a group in the summary."""
        m = _import_module()
        df, features = self._make_labeled_df()

        summary = m.compute_cluster_summary(df, features, label_col="hdbscan_label")

        label_col = summary.columns[0]
        labels_in_summary = set(summary[label_col].astype(int).tolist())
        assert -1 in labels_in_summary, (
            "Noise group (label=-1) must be included in cluster summary."
        )

    def test_compute_cluster_summary_mean_matches_pandas(self):
        """Mean values must match those computed directly by pandas groupby.

        This is a spot-check: if the aggregation is wrong the means will differ.
        """
        m = _import_module()
        rng = np.random.default_rng(7)
        n = 30
        df = pd.DataFrame({
            "call_length_s": rng.uniform(0.01, 0.5, n),
            "slope": rng.uniform(-100, 100, n),
            "hdbscan_label": np.array([0] * 15 + [1] * 15),
        })
        features = ["call_length_s", "slope"]

        summary = m.compute_cluster_summary(df, features, label_col="hdbscan_label")

        # Direct pandas reference computation
        grp = df.groupby("hdbscan_label")
        for lbl in [0, 1]:
            expected_mean_slope = grp.get_group(lbl)["slope"].mean()
            # Find the slope_mean column in summary
            slope_mean_col = next(
                c for c in summary.columns if "slope" in c and "mean" in c
            )
            label_col = summary.columns[0]
            actual = summary.loc[
                summary[label_col].astype(int) == lbl, slope_mean_col
            ].values[0]
            np.testing.assert_allclose(
                actual, expected_mean_slope, rtol=1e-10,
                err_msg=f"Mean slope for label {lbl} does not match pandas reference.",
            )


# ===========================================================================
# run_umap tests
# ===========================================================================

class TestRunUmap:
    """Tests for run_umap(X_scaled, n_components, n_neighbors, min_dist, seed)."""

    def test_run_umap_output_shape(self):
        """Output shape must be (N, n_components)."""
        m = _import_module()
        rng = np.random.default_rng(0)
        X = rng.standard_normal((80, 10)).astype(np.float32)

        emb = m.run_umap(X, n_components=2, n_neighbors=15, min_dist=0.1, seed=42)

        assert emb.shape == (80, 2), f"Expected (80, 2), got {emb.shape}."

    def test_run_umap_2d_and_8d_shapes(self):
        """Both 2D (visualization) and 8D (HDBSCAN input) embeddings must work."""
        m = _import_module()
        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 10)).astype(np.float32)

        emb_2d = m.run_umap(X, n_components=2, n_neighbors=10, min_dist=0.1, seed=42)
        emb_8d = m.run_umap(X, n_components=8, n_neighbors=10, min_dist=0.1, seed=42)

        assert emb_2d.shape == (60, 2), f"2D UMAP: expected (60, 2), got {emb_2d.shape}."
        assert emb_8d.shape == (60, 8), f"8D UMAP: expected (60, 8), got {emb_8d.shape}."

    def test_run_umap_output_is_float(self):
        """Embedding values must be finite floating-point numbers."""
        m = _import_module()
        rng = np.random.default_rng(5)
        X = rng.standard_normal((50, 5)).astype(np.float32)

        emb = m.run_umap(X, n_components=2, n_neighbors=10, min_dist=0.1, seed=42)

        assert emb.dtype.kind == "f", f"Expected float dtype, got {emb.dtype}."
        assert np.all(np.isfinite(emb)), "UMAP output must contain only finite values."

    def test_run_umap_reproducible_with_same_seed(self):
        """Same seed must produce identical embeddings."""
        m = _import_module()
        rng = np.random.default_rng(1)
        X = rng.standard_normal((50, 6)).astype(np.float32)

        emb1 = m.run_umap(X, n_components=2, n_neighbors=10, min_dist=0.1, seed=99)
        emb2 = m.run_umap(X, n_components=2, n_neighbors=10, min_dist=0.1, seed=99)

        np.testing.assert_array_equal(emb1, emb2, err_msg="UMAP must be reproducible with same seed.")


# ===========================================================================
# run_hdbscan tests
# ===========================================================================

class TestRunHdbscan:
    """Tests for run_hdbscan(embedding, min_cluster_size, min_samples)."""

    def test_run_hdbscan_output_shapes_and_types(self):
        """Labels and probabilities must have shape (N,) with correct dtypes."""
        m = _import_module()
        rng = np.random.default_rng(0)
        # Two clear clusters
        X = np.vstack([
            rng.standard_normal((50, 4)) + np.array([5, 0, 0, 0]),
            rng.standard_normal((50, 4)) - np.array([5, 0, 0, 0]),
        ]).astype(np.float32)

        labels, probs = m.run_hdbscan(X, min_cluster_size=5, min_samples=2)

        assert labels.shape == (100,), f"Labels shape must be (100,), got {labels.shape}."
        assert probs.shape == (100,), f"Probabilities shape must be (100,), got {probs.shape}."
        assert labels.dtype.kind in ("i", "u"), f"Labels must be integer dtype, got {labels.dtype}."
        assert probs.dtype.kind == "f", f"Probabilities must be float dtype, got {probs.dtype}."

    def test_run_hdbscan_probabilities_in_range(self):
        """All probabilities must lie in [0.0, 1.0]."""
        m = _import_module()
        rng = np.random.default_rng(2)
        X = rng.standard_normal((80, 3)).astype(np.float32)

        _, probs = m.run_hdbscan(X, min_cluster_size=5, min_samples=2)

        assert np.all(probs >= 0.0), f"Min probability is {probs.min()}, expected >= 0."
        assert np.all(probs <= 1.0), f"Max probability is {probs.max()}, expected <= 1."

    def test_run_hdbscan_labels_contain_minus_one_for_noise(self):
        """HDBSCAN must produce -1 labels for noise/outlier points.

        With high min_cluster_size on small data, many points should be noise.
        """
        m = _import_module()
        rng = np.random.default_rng(3)
        # Sparse random data — with large min_cluster_size most will be noise
        X = rng.standard_normal((30, 2)).astype(np.float32)

        labels, _ = m.run_hdbscan(X, min_cluster_size=20, min_samples=5)

        assert -1 in labels, (
            "Expected noise points (label=-1) when min_cluster_size is large relative to N."
        )

    def test_run_hdbscan_all_noise_case(self):
        """When every point is noise (label=-1), output must still be valid arrays.

        Edge case spec: 'all points become noise in HDBSCAN'.
        """
        m = _import_module()
        rng = np.random.default_rng(4)
        # Fully random sparse data with min_cluster_size larger than any plausible cluster
        X = rng.standard_normal((20, 2)).astype(np.float32)

        labels, probs = m.run_hdbscan(X, min_cluster_size=19, min_samples=15)

        assert labels.shape == (20,)
        assert probs.shape == (20,)
        # Either all noise or almost all noise — the function must not crash
        assert np.all(labels >= -1), "Labels must be >= -1 (no invalid values)."
        assert np.all(probs >= 0.0)


# ===========================================================================
# plot function tests
# ===========================================================================

class TestPlotFunctions:
    """Tests for plot_umap_scatter and plot_contingency_matrix."""

    def test_plot_umap_scatter_creates_png(self, tmp_path):
        """plot_umap_scatter must write a PNG file at output_path."""
        m = _import_module()
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((50, 2))
        labels = np.array([0] * 20 + [1] * 15 + [-1] * 15)
        out = tmp_path / "scatter.png"

        m.plot_umap_scatter(emb, labels, out, title="Test Scatter")

        assert out.exists(), f"PNG was not created at {out}."
        assert out.stat().st_size > 1000, "PNG file appears empty (< 1 KB)."

    def test_plot_umap_scatter_creates_parent_dirs(self, tmp_path):
        """plot_umap_scatter must create parent directories if they do not exist."""
        m = _import_module()
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((30, 2))
        labels = np.array([0] * 15 + [-1] * 15)
        # Deep nested path that doesn't exist yet
        out = tmp_path / "nested" / "deep" / "scatter.png"

        m.plot_umap_scatter(emb, labels, out, title="Nested Dir Test")

        assert out.exists(), "PNG must be created even when parent dirs are missing."

    def test_plot_umap_scatter_all_noise(self, tmp_path):
        """plot_umap_scatter must not crash when all points are noise (-1)."""
        m = _import_module()
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((20, 2))
        labels = np.full(20, -1, dtype=int)
        out = tmp_path / "all_noise.png"

        m.plot_umap_scatter(emb, labels, out, title="All Noise")

        assert out.exists(), "PNG must be created even when all labels are -1."

    def test_plot_contingency_matrix_creates_png(self, tmp_path):
        """plot_contingency_matrix must write a PNG file at output_path."""
        m = _import_module()
        rng = np.random.default_rng(0)
        old_labels = rng.integers(0, 27, 100)
        new_labels = rng.integers(-1, 5, 100)
        out = tmp_path / "contingency.png"

        m.plot_contingency_matrix(old_labels, new_labels, out)

        assert out.exists(), f"Contingency matrix PNG was not created at {out}."
        assert out.stat().st_size > 1000, "PNG file appears empty (< 1 KB)."


# ===========================================================================
# Integration test
# ===========================================================================

class TestIntegration:
    """End-to-end integration test through the full pipeline on tiny synthetic data."""

    def test_integration_small_dataset_full_pipeline(self, tmp_path):
        """Full pipeline from CSV to output DataFrame must work on small synthetic data.

        This test actually runs UMAP and HDBSCAN (not mocked) to catch integration bugs.
        Spec requirement: 'at least one integration test that actually runs them on tiny data'.
        """
        m = _import_module()

        # Build a CSV with 80 valid rows and 5 NaN rows
        csv = tmp_path / "integration.csv"
        df_orig = _write_synthetic_csv(csv, n_valid=80, n_nan=5, seed=123)

        # Step 1: load and prepare
        df_full, df_valid, X_scaled, features = m.load_and_prepare(
            str(csv), exclude_features=[]
        )
        assert len(df_full) == 85
        assert len(df_valid) == 80
        assert X_scaled.shape == (80, 10)

        # Step 2: UMAP 2D (visualization)
        emb_2d = m.run_umap(X_scaled, n_components=2, n_neighbors=10, min_dist=0.1, seed=42)
        assert emb_2d.shape == (80, 2)

        # Step 3: UMAP ND (HDBSCAN input) — use 3D for speed
        emb_nd = m.run_umap(X_scaled, n_components=3, n_neighbors=10, min_dist=0.1, seed=42)
        assert emb_nd.shape == (80, 3)

        # Step 4: HDBSCAN on ND embedding
        labels, probs = m.run_hdbscan(emb_nd, min_cluster_size=5, min_samples=2)
        assert labels.shape == (80,)
        assert probs.shape == (80,)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

        # Step 5: assemble output
        df_out = m.build_output_df(df_full, df_valid, emb_2d, labels, probs)
        assert len(df_out) == 85, f"Output row count must match input: {len(df_out)} != 85."

        nan_indices = df_full.index.difference(df_valid.index)
        assert (df_out.loc[nan_indices, "hdbscan_label"] == -1).all()
        assert (df_out.loc[nan_indices, "hdbscan_probability"] == 0.0).all()

        # Step 6: cluster summary
        df_valid_labeled = df_valid.copy()
        df_valid_labeled["hdbscan_label"] = labels
        summary = m.compute_cluster_summary(df_valid_labeled, features, label_col="hdbscan_label")
        assert "count" in summary.columns
        assert len(summary) >= 1, "Summary must have at least one row."

        # Step 7: scatter plot
        out_scatter = tmp_path / "scatter.png"
        m.plot_umap_scatter(emb_2d, labels, out_scatter, title="Integration Test")
        assert out_scatter.exists() and out_scatter.stat().st_size > 1000

    def test_integration_output_csv_column_count(self, tmp_path):
        """Output DataFrame must have original columns plus exactly 4 new ones."""
        m = _import_module()
        csv = tmp_path / "cols.csv"
        df_orig = _write_synthetic_csv(csv, n_valid=60, n_nan=3)
        original_col_count = len(df_orig.columns)

        df_full, df_valid, X_scaled, features = m.load_and_prepare(
            str(csv), exclude_features=[]
        )
        emb_2d = m.run_umap(X_scaled, n_components=2, n_neighbors=10, min_dist=0.1, seed=0)
        labels, probs = m.run_hdbscan(emb_2d, min_cluster_size=5, min_samples=2)
        df_out = m.build_output_df(df_full, df_valid, emb_2d, labels, probs)

        expected_col_count = original_col_count + 4
        assert len(df_out.columns) == expected_col_count, (
            f"Expected {expected_col_count} columns "
            f"({original_col_count} original + 4 new), got {len(df_out.columns)}."
        )
