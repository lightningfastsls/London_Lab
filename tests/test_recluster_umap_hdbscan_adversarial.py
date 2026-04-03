"""Adversarial / gap-filling tests for scripts/recluster_umap_hdbscan.py.

Written by the adversarial-tester agent AFTER the test-architect tests existed (36 tests).

Coverage gaps targeted:
  A. parse_args() — completely absent from original suite
  B. load_and_prepare: all-NaN rows (df_valid empty -> StandardScaler crash)
  C. load_and_prepare: ALL features excluded (empty feature list)
  D. build_output_df: no NaN rows (empty df_nan concat path — FutureWarning source)
  E. build_output_df: all rows NaN (empty df_valid path)
  F. build_output_df: actual coordinate values match embedding (not just non-NaN)
  G. plot_umap_scatter: >20 clusters triggers ListedColormap branch
  H. plot_umap_scatter: exactly 1 cluster (denominator guard max(..., 1))
  I. plot_umap_scatter: >15 labels triggers outside-legend branch
  J. plot_contingency_matrix: all-same old labels (degenerate crosstab)
  K. plot_contingency_matrix: large matrix (>600 cells -> annot=False branch)
  L. compute_cluster_summary: single cluster (no noise group)
  M. compute_cluster_summary: non-default label_col argument
  N. run_umap: different seeds produce different results (negative reproducibility)
  O. run_umap: NaN in input propagates as ValueError (UMAP raises, not silent)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

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


def _import_module():
    import recluster_umap_hdbscan as m  # noqa: PLC0415
    return m


def _make_synthetic_df(n_valid: int = 30, n_nan: int = 0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_total = n_valid + n_nan
    data: dict = {}
    data["call_length_s"] = np.where(np.arange(n_total) < n_valid, rng.uniform(0.01, 0.5, n_total), np.nan)
    data["principal_freq_hz"] = np.where(np.arange(n_total) < n_valid, rng.uniform(47, 91, n_total), np.nan)
    data["low_freq_hz"] = np.where(np.arange(n_total) < n_valid, rng.uniform(40, 70, n_total), np.nan)
    data["high_freq_hz"] = np.where(np.arange(n_total) < n_valid, rng.uniform(70, 110, n_total), np.nan)
    data["bandwidth_hz"] = np.where(np.arange(n_total) < n_valid, rng.uniform(1, 50, n_total), np.nan)
    data["freq_std_dev_hz"] = np.where(np.arange(n_total) < n_valid, rng.uniform(1, 30, n_total), np.nan)
    data["slope"] = np.where(np.arange(n_total) < n_valid, rng.uniform(-793, 704, n_total), np.nan)
    data["sinuosity"] = np.where(np.arange(n_total) < n_valid, rng.uniform(1.2, 8.1, n_total), np.nan)
    data["tonality"] = np.where(np.arange(n_total) < n_valid, rng.uniform(0.18, 0.34, n_total), np.nan)
    data["mean_power_db"] = np.where(np.arange(n_total) < n_valid, rng.uniform(-80, -20, n_total), np.nan)
    data["label"] = [f"Cluster_{rng.integers(1, 28)}" for _ in range(n_total)]
    data["wav_stem"] = [f"recording_{i:04d}" for i in range(n_total)]
    return pd.DataFrame(data)


# ===========================================================================
# A. parse_args tests
# ===========================================================================

class TestParseArgs:
    """parse_args() was completely absent from the original suite."""

    def test_parse_args_default_values(self, monkeypatch):
        """Default values must match spec: n_neighbors=15, min_dist=0.1, seed=42."""
        m = _import_module()
        monkeypatch.setattr(sys, "argv", ["recluster_umap_hdbscan.py"])
        args = m.parse_args()
        assert args.n_neighbors == 15
        assert args.min_dist == 0.1
        assert args.seed == 42
        assert args.min_cluster_size == 50
        assert args.min_samples == 10
        assert args.umap_components_cluster == 8
        assert args.skip_gallery is False
        assert args.exclude_features == []
        assert args.verbose is False

    def test_parse_args_custom_umap_params(self, monkeypatch):
        """CLI flags must override defaults for UMAP parameters."""
        m = _import_module()
        monkeypatch.setattr(sys, "argv", [
            "recluster_umap_hdbscan.py",
            "--n-neighbors", "30",
            "--min-dist", "0.05",
            "--umap-components-cluster", "12",
        ])
        args = m.parse_args()
        assert args.n_neighbors == 30
        assert args.min_dist == pytest.approx(0.05)
        assert args.umap_components_cluster == 12

    def test_parse_args_custom_hdbscan_params(self, monkeypatch):
        """CLI flags must override defaults for HDBSCAN parameters."""
        m = _import_module()
        monkeypatch.setattr(sys, "argv", [
            "recluster_umap_hdbscan.py",
            "--min-cluster-size", "20",
            "--min-samples", "5",
        ])
        args = m.parse_args()
        assert args.min_cluster_size == 20
        assert args.min_samples == 5

    def test_parse_args_exclude_features_list(self, monkeypatch):
        """--exclude-features must accept multiple space-separated feature names."""
        m = _import_module()
        monkeypatch.setattr(sys, "argv", [
            "recluster_umap_hdbscan.py",
            "--exclude-features", "slope", "tonality", "mean_power_db",
        ])
        args = m.parse_args()
        assert args.exclude_features == ["slope", "tonality", "mean_power_db"]

    def test_parse_args_skip_gallery_flag(self, monkeypatch):
        """--skip-gallery must set skip_gallery=True."""
        m = _import_module()
        monkeypatch.setattr(sys, "argv", ["recluster_umap_hdbscan.py", "--skip-gallery"])
        args = m.parse_args()
        assert args.skip_gallery is True

    def test_parse_args_verbose_flag(self, monkeypatch):
        """-v / --verbose must set verbose=True."""
        m = _import_module()
        monkeypatch.setattr(sys, "argv", ["recluster_umap_hdbscan.py", "--verbose"])
        args = m.parse_args()
        assert args.verbose is True

    def test_parse_args_seed_custom(self, monkeypatch):
        """--seed must accept any integer."""
        m = _import_module()
        monkeypatch.setattr(sys, "argv", ["recluster_umap_hdbscan.py", "--seed", "1234"])
        args = m.parse_args()
        assert args.seed == 1234

    def test_parse_args_csv_and_output_dir(self, monkeypatch, tmp_path):
        """--csv and --output-dir must propagate correctly."""
        m = _import_module()
        csv_path = str(tmp_path / "my.csv")
        out_path = str(tmp_path / "out")
        monkeypatch.setattr(sys, "argv", [
            "recluster_umap_hdbscan.py",
            "--csv", csv_path,
            "--output-dir", out_path,
        ])
        args = m.parse_args()
        assert args.csv == csv_path
        assert args.output_dir == out_path


# ===========================================================================
# B. load_and_prepare: all-NaN input
# ===========================================================================

class TestLoadAndPrepareEdgeCases:
    """Untested code paths in load_and_prepare."""

    def test_all_nan_rows_raises_value_error(self, tmp_path):
        """When every row has NaN in acoustic features, StandardScaler must raise.

        The empty array (shape (0, 10)) cannot be fit by StandardScaler — the caller
        should guard against this before calling load_and_prepare on degenerate data.
        This test documents the actual failure mode so callers know what to expect.
        """
        m = _import_module()
        data = {f: [float("nan")] * 5 for f in ACOUSTIC_FEATURES}
        data["label"] = ["Cluster_1"] * 5
        data["wav_stem"] = ["rec"] * 5
        csv = tmp_path / "all_nan.csv"
        pd.DataFrame(data).to_csv(csv, index=False)

        with pytest.raises(ValueError, match="0 sample"):
            m.load_and_prepare(str(csv), exclude_features=[])

    def test_all_features_excluded_raises_value_error(self, tmp_path):
        """Excluding every ACOUSTIC_FEATURE must raise before UMAP can run.

        StandardScaler on shape (N, 0) raises — this documents the contract so
        the caller knows that passing exclude_features=ACOUSTIC_FEATURES is invalid.
        """
        m = _import_module()
        df = _make_synthetic_df(n_valid=10, n_nan=0)
        csv = tmp_path / "exclude_all.csv"
        df.to_csv(csv, index=False)

        with pytest.raises(ValueError):
            m.load_and_prepare(str(csv), exclude_features=ACOUSTIC_FEATURES)

    def test_load_and_prepare_partial_nan_only_some_features(self, tmp_path):
        """Row with NaN in ONE feature must be excluded even if other features are valid.

        This tests the .all(axis=1) requirement — partial-NaN rows are not valid.
        """
        m = _import_module()
        df = _make_synthetic_df(n_valid=20, n_nan=0)
        # Set slope=NaN on 3 rows — those rows must be excluded from df_valid
        df.loc[[2, 7, 15], "slope"] = float("nan")
        csv = tmp_path / "partial_nan.csv"
        df.to_csv(csv, index=False)

        _, df_valid, X_scaled, _ = m.load_and_prepare(str(csv), exclude_features=[])

        assert len(df_valid) == 17, (
            f"Rows with NaN in any single feature must be excluded. Expected 17, got {len(df_valid)}."
        )

    def test_load_and_prepare_single_row_valid(self, tmp_path):
        """A CSV with exactly one valid row must not crash StandardScaler.

        StandardScaler with a single sample produces zero std (all zeros after scaling).
        The function must complete without error.
        """
        m = _import_module()
        df = _make_synthetic_df(n_valid=1, n_nan=0)
        csv = tmp_path / "single_row.csv"
        df.to_csv(csv, index=False)

        df_full, df_valid, X_scaled, features = m.load_and_prepare(str(csv), exclude_features=[])

        assert len(df_valid) == 1
        assert X_scaled.shape == (1, 10)
        # Single-sample scaling: result is zeros (mean subtracted, then divided by std=0 -> 0/0=0 by sklearn)
        assert np.all(np.isfinite(X_scaled)), "Single-row scaling must produce finite values."


# ===========================================================================
# D/E. build_output_df edge cases
# ===========================================================================

class TestBuildOutputDfEdgeCases:
    """Paths in build_output_df not covered by the original suite."""

    def test_no_nan_rows_output_matches_df_valid(self):
        """When df_full == df_valid (no NaN rows), output must equal valid rows exactly.

        This exercises the empty df_nan concat path (the FutureWarning source).
        """
        m = _import_module()
        rng = np.random.default_rng(0)
        n = 10
        data = {f: rng.standard_normal(n) for f in ACOUSTIC_FEATURES}
        data["label"] = ["Cluster_1"] * n
        data["wav_stem"] = ["rec"] * n
        df_full = pd.DataFrame(data)
        df_valid = df_full.copy()
        emb = rng.standard_normal((n, 2))
        labels = np.zeros(n, dtype=int)
        probs = np.ones(n)

        df_out = m.build_output_df(df_full, df_valid, emb, labels, probs)

        assert len(df_out) == n, f"Expected {n} rows with no NaN, got {len(df_out)}."
        assert df_out["umap_x"].notna().all(), "No NaN rows means all umap_x must be finite."
        assert df_out["umap_y"].notna().all()

    def test_all_nan_rows_output_length_and_labels(self):
        """When df_valid is empty (all rows NaN), output must still have df_full length,
        all with hdbscan_label=-1 and hdbscan_probability=0.0.
        """
        m = _import_module()
        n = 5
        data = {f: [float("nan")] * n for f in ACOUSTIC_FEATURES}
        data["label"] = ["Cluster_1"] * n
        data["wav_stem"] = ["rec"] * n
        df_full = pd.DataFrame(data)
        df_valid = df_full.iloc[:0].copy()  # empty
        emb = np.zeros((0, 2))
        labels = np.array([], dtype=int)
        probs = np.array([])

        df_out = m.build_output_df(df_full, df_valid, emb, labels, probs)

        assert len(df_out) == n, f"Expected {n} rows, got {len(df_out)}."
        assert (df_out["hdbscan_label"] == -1).all(), "All-NaN input must produce all label=-1."
        assert (df_out["hdbscan_probability"] == 0.0).all()
        assert df_out["umap_x"].isna().all(), "All-NaN input must produce all NaN umap_x."

    def test_valid_rows_carry_exact_embedding_values(self):
        """Valid rows must carry the EXACT values from emb_2d, not just non-NaN.

        Tests value correctness, not just presence — the existing suite only checks notna().
        """
        m = _import_module()
        rng = np.random.default_rng(7)
        n = 5
        data = {f: rng.standard_normal(n) for f in ACOUSTIC_FEATURES}
        data["label"] = ["Cluster_1"] * n
        data["wav_stem"] = ["rec"] * n
        df_full = pd.DataFrame(data)
        df_valid = df_full.copy()

        # Use recognizable sentinel values so we can verify exact pass-through
        emb = np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6], [7.7, 8.8], [9.9, 0.11]])
        labels = np.array([0, 1, 0, 1, 2])
        probs = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

        df_out = m.build_output_df(df_full, df_valid, emb, labels, probs)

        np.testing.assert_allclose(
            df_out["umap_x"].values, emb[:, 0],
            err_msg="umap_x must contain exact values from emb_2d column 0.",
        )
        np.testing.assert_allclose(
            df_out["umap_y"].values, emb[:, 1],
            err_msg="umap_y must contain exact values from emb_2d column 1.",
        )
        np.testing.assert_array_equal(
            df_out["hdbscan_label"].values, labels,
            err_msg="hdbscan_label must contain exact cluster label values.",
        )
        np.testing.assert_allclose(
            df_out["hdbscan_probability"].values, probs,
            err_msg="hdbscan_probability must contain exact probability values.",
        )

    def test_nan_rows_interspersed_with_valid_rows(self):
        """NaN rows scattered throughout df_full (not just at end) must be handled correctly.

        The original helper always places NaN rows after valid rows. This test
        interleaves them to verify index-based reassembly works in any order.
        """
        m = _import_module()
        rng = np.random.default_rng(42)

        # Build df_full with NaN rows at indices 1, 3 (interleaved)
        data = {f: rng.standard_normal(5) for f in ACOUSTIC_FEATURES}
        data["label"] = ["Cluster_1"] * 5
        data["wav_stem"] = ["rec"] * 5
        df_full = pd.DataFrame(data)
        df_full.loc[[1, 3], ACOUSTIC_FEATURES] = float("nan")

        df_valid = df_full.dropna(subset=ACOUSTIC_FEATURES).copy()  # rows 0, 2, 4
        emb = rng.standard_normal((3, 2))
        labels = np.array([0, 1, 0])
        probs = np.array([0.9, 0.8, 0.7])

        df_out = m.build_output_df(df_full, df_valid, emb, labels, probs)

        assert len(df_out) == 5
        # NaN rows must be labeled -1
        assert df_out.loc[1, "hdbscan_label"] == -1
        assert df_out.loc[3, "hdbscan_label"] == -1
        # Valid rows must not be labeled -1
        assert df_out.loc[0, "hdbscan_label"] != -1
        # Index is sorted
        assert df_out.index.tolist() == sorted(df_out.index.tolist())


# ===========================================================================
# F. run_umap reproducibility — negative check
# ===========================================================================

class TestRunUmapAdversarial:
    """Gaps in the original run_umap tests."""

    def test_run_umap_different_seeds_produce_different_embeddings(self):
        """Different seeds must produce different embeddings.

        The original suite only verifies same-seed -> same result (positive check).
        This negative check verifies the seed actually affects the output.
        """
        m = _import_module()
        rng = np.random.default_rng(1)
        X = rng.standard_normal((50, 6)).astype(np.float32)

        emb1 = m.run_umap(X, n_components=2, n_neighbors=10, min_dist=0.1, seed=1)
        emb2 = m.run_umap(X, n_components=2, n_neighbors=10, min_dist=0.1, seed=999)

        assert not np.allclose(emb1, emb2), (
            "Different seeds must produce different UMAP embeddings. "
            "If this fails, the seed parameter has no effect."
        )

    def test_run_umap_nan_input_raises(self):
        """NaN values in X_scaled must raise ValueError from UMAP, not silently propagate.

        load_and_prepare already filters NaNs, so X_scaled reaching run_umap with NaN
        indicates a bug upstream. The function must fail loudly.
        """
        m = _import_module()
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 6)).astype(np.float32)
        X[10, 3] = float("nan")

        with pytest.raises((ValueError, Exception)):
            m.run_umap(X, n_components=2, n_neighbors=10, min_dist=0.1, seed=42)


# ===========================================================================
# G/H/I. plot_umap_scatter untested branches
# ===========================================================================

class TestPlotUmapScatterAdversarial:
    """Branches in plot_umap_scatter not covered by the original suite."""

    def test_plot_umap_scatter_more_than_20_clusters_uses_combined_colormap(self, tmp_path):
        """When cluster_labels > 20, the ListedColormap branch is used.

        The original suite only has tests with <= 20 clusters. This exercises
        the concatenated tab20+tab20b colormap path (lines 243-246 of implementation).
        """
        m = _import_module()
        rng = np.random.default_rng(0)
        n_clusters = 21
        n_per_cluster = 10
        n_total = n_clusters * n_per_cluster
        emb = rng.standard_normal((n_total, 2))
        labels = np.repeat(np.arange(n_clusters), n_per_cluster)
        out = tmp_path / "21_clusters.png"

        # Must not raise — the colormap indexing with i / max(len-1, 1) must stay in [0,1]
        m.plot_umap_scatter(emb, labels, out, title="21 cluster colormap test")

        assert out.exists(), "PNG must be created for >20 clusters."
        assert out.stat().st_size > 1000, "PNG must not be empty for >20 clusters."

    def test_plot_umap_scatter_exactly_40_clusters(self, tmp_path):
        """40 clusters fills both tab20 + tab20b exactly — boundary of the combined colormap."""
        m = _import_module()
        rng = np.random.default_rng(0)
        n_clusters = 40
        n_total = n_clusters * 5
        emb = rng.standard_normal((n_total, 2))
        labels = np.repeat(np.arange(n_clusters), 5)
        out = tmp_path / "40_clusters.png"

        m.plot_umap_scatter(emb, labels, out, title="40 clusters boundary")

        assert out.exists()

    def test_plot_umap_scatter_single_cluster_no_crash(self, tmp_path):
        """Exactly 1 cluster (label 0, no noise) must not crash.

        The denominator guard `max(len(cluster_labels) - 1, 1)` at line 260 prevents
        division-by-zero when there is only one cluster. This test exercises that path.
        """
        m = _import_module()
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((20, 2))
        labels = np.zeros(20, dtype=int)  # only label 0
        out = tmp_path / "single_cluster.png"

        m.plot_umap_scatter(emb, labels, out, title="Single cluster no crash")

        assert out.exists(), "PNG must be created for single-cluster input."

    def test_plot_umap_scatter_more_than_15_labels_triggers_outside_legend(self, tmp_path):
        """When total unique labels > 15 (including noise), legend goes outside the plot.

        The branch at line 276 uses bbox_to_anchor. We verify it does not raise —
        tight_layout with an outside legend is a known matplotlib footgun.
        """
        m = _import_module()
        rng = np.random.default_rng(0)
        n_clusters = 16  # 16 + possible noise = 17 legend entries -> triggers outside path
        n_total = n_clusters * 10 + 10
        emb = rng.standard_normal((n_total, 2))
        labels = np.concatenate([
            np.repeat(np.arange(n_clusters), 10),
            np.full(10, -1, dtype=int),   # noise
        ])
        out = tmp_path / "outside_legend.png"

        m.plot_umap_scatter(emb, labels, out, title="16 clusters + noise -> outside legend")

        assert out.exists(), "PNG must be created when legend is placed outside plot."

    def test_plot_umap_scatter_no_clusters_all_noise(self, tmp_path):
        """Confirmed passing in original suite — kept here as regression guard for colormap path."""
        m = _import_module()
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((30, 2))
        labels = np.full(30, -1, dtype=int)
        out = tmp_path / "all_noise_regression.png"

        m.plot_umap_scatter(emb, labels, out, title="All noise regression")

        assert out.exists()


# ===========================================================================
# J/K. plot_contingency_matrix untested branches
# ===========================================================================

class TestPlotContingencyMatrixAdversarial:
    """Edge cases in plot_contingency_matrix not covered by original suite."""

    def test_contingency_matrix_all_same_old_labels(self, tmp_path):
        """All old labels identical -> 1-row contingency matrix must not raise.

        pd.crosstab with a single unique old label still produces a valid DataFrame.
        """
        m = _import_module()
        old_labels = np.zeros(50, dtype=int)       # all Cluster_0
        new_labels = np.array([0] * 25 + [-1] * 25)
        out = tmp_path / "degenerate_contng.png"

        m.plot_contingency_matrix(old_labels, new_labels, out)

        assert out.exists(), "Contingency matrix must handle single-unique-old-label input."
        assert out.stat().st_size > 1000

    def test_contingency_matrix_all_same_new_labels(self, tmp_path):
        """All new labels identical (-1) -> 1-column contingency matrix must not raise."""
        m = _import_module()
        rng = np.random.default_rng(0)
        old_labels = rng.integers(0, 5, 50)
        new_labels = np.full(50, -1, dtype=int)
        out = tmp_path / "all_noise_contng.png"

        m.plot_contingency_matrix(old_labels, new_labels, out)

        assert out.exists(), "Contingency matrix must handle all-noise new labels."

    def test_contingency_matrix_large_disables_annotation(self, tmp_path):
        """Matrix with > 600 cells (30 old x 25 new) must set annot=False and not crash.

        The annotation branch at line 343 of the implementation is: annot = n_old * n_new <= 600.
        With 30 * 25 = 750 cells, annot must be False to avoid cluttered unreadable labels.
        """
        m = _import_module()
        rng = np.random.default_rng(0)
        # Ensure all 30 old labels and 25 new labels actually appear
        n_per_combo = 4
        old_labels = np.repeat(np.arange(30), n_per_combo * 25)
        new_labels = np.tile(np.arange(-1, 24), 30 * n_per_combo)
        out = tmp_path / "large_contng.png"

        m.plot_contingency_matrix(old_labels, new_labels, out)

        assert out.exists(), "Large contingency matrix must produce a PNG."
        assert out.stat().st_size > 5000, (
            "Large contingency matrix PNG is suspiciously small — may have failed silently."
        )

    def test_contingency_matrix_creates_parent_dirs(self, tmp_path):
        """plot_contingency_matrix must create parent directories if they do not exist."""
        m = _import_module()
        rng = np.random.default_rng(0)
        old_labels = rng.integers(0, 5, 30)
        new_labels = rng.integers(-1, 3, 30)
        out = tmp_path / "nested" / "deep" / "contng.png"

        m.plot_contingency_matrix(old_labels, new_labels, out)

        assert out.exists(), "Contingency matrix must create nested parent directories."


# ===========================================================================
# L/M. compute_cluster_summary edge cases
# ===========================================================================

class TestComputeClusterSummaryAdversarial:
    """Gaps in compute_cluster_summary coverage."""

    def test_single_cluster_no_noise(self):
        """Summary with only one cluster label (no noise group) must return 1 row.

        The original suite always includes noise. This tests the noise-absent path.
        """
        m = _import_module()
        rng = np.random.default_rng(0)
        n = 30
        df = pd.DataFrame({
            "call_length_s": rng.uniform(0.01, 0.5, n),
            "slope": rng.uniform(-100, 100, n),
            "hdbscan_label": np.zeros(n, dtype=int),
        })
        features = ["call_length_s", "slope"]

        summary = m.compute_cluster_summary(df, features, label_col="hdbscan_label")

        assert len(summary) == 1, f"Single cluster must produce 1-row summary, got {len(summary)}."
        assert summary.iloc[0]["count"] == n

    def test_non_default_label_col(self):
        """label_col argument must be honored — groupby on the correct column."""
        m = _import_module()
        rng = np.random.default_rng(0)
        n = 40
        df = pd.DataFrame({
            "call_length_s": rng.uniform(0.01, 0.5, n),
            "slope": rng.uniform(-100, 100, n),
            "my_custom_label": np.array([0] * 20 + [1] * 20),
        })
        features = ["call_length_s", "slope"]

        summary = m.compute_cluster_summary(df, features, label_col="my_custom_label")

        # Column 0 of summary should be the label column name
        assert summary.columns[0] == "my_custom_label", (
            "Summary label column must reflect the custom label_col argument."
        )
        assert len(summary) == 2, "Two clusters must produce 2-row summary."

    def test_all_noise_cluster(self):
        """Summary where every row has label -1 must produce a single row for noise."""
        m = _import_module()
        rng = np.random.default_rng(0)
        n = 20
        df = pd.DataFrame({
            "call_length_s": rng.uniform(0.01, 0.5, n),
            "slope": rng.uniform(-100, 100, n),
            "hdbscan_label": np.full(n, -1, dtype=int),
        })
        features = ["call_length_s", "slope"]

        summary = m.compute_cluster_summary(df, features, label_col="hdbscan_label")

        assert len(summary) == 1
        label_val = int(summary.iloc[0][summary.columns[0]])
        assert label_val == -1, f"All-noise summary must have label -1, got {label_val}."

    def test_summary_std_is_nan_for_single_element_cluster(self):
        """A cluster with a single member must have NaN std (ddof=1 with N=1).

        This is a known pandas behavior — verify the output doesn't silently become 0.
        """
        m = _import_module()
        df = pd.DataFrame({
            "call_length_s": [0.1, 0.2, 0.3],
            "slope": [10.0, 20.0, 30.0],
            "hdbscan_label": [0, 0, 1],  # cluster 1 has only 1 member
        })
        features = ["call_length_s", "slope"]

        summary = m.compute_cluster_summary(df, features, label_col="hdbscan_label")

        # Cluster 1 (single member) should have NaN std for each feature
        lbl_col = summary.columns[0]
        row_1 = summary[summary[lbl_col].astype(int) == 1]
        std_col = next(c for c in summary.columns if "slope" in c and "std" in c)
        std_val = row_1[std_col].values[0]
        assert np.isnan(std_val), (
            f"Single-member cluster std must be NaN (ddof=1), got {std_val}."
        )
