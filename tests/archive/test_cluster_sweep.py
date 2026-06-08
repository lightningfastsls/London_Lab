"""Tests for cluster_sweep — written by test-architect BEFORE implementation.

Module under test: src/usv_spectrogram/classification/cluster_sweep.py
Spec source: ROADMAP_SIS_BENCHMARK.md module 17.7 (lines 649-720)

ROADMAP test plan coverage:
  1. k=3 has highest silhouette on 3-blob synthetic data        -> test_best_k_has_highest_silhouette_on_blob_data
  2. Output dict has one entry per k in cfg.k_values            -> test_output_keys_match_k_values
  3. Each entry has labels of length N, centers shape (k, D)    -> test_output_shapes_per_k
  4. Inertia non-increasing as k increases                      -> test_inertia_monotonically_non_increasing
  5. Silhouette subsample matches full within 0.05              -> test_silhouette_subsample_close_to_full
  6. Empty feature matrix raises ValueError                     -> test_empty_features_raises_value_error
  7. N < max(k_values) gracefully skipped (warn, don't crash)   -> test_small_n_skips_large_k_with_warning
  8. Reproducibility: same random_state -> same labels           -> test_same_random_state_produces_same_labels

Additional coverage (recurring gap patterns):
  - Config defaults match spec values                           -> test_cluster_sweep_config_defaults
  - Config is frozen (immutable)                                -> test_cluster_sweep_config_is_frozen
  - Output dict values contain all required keys                -> test_each_result_entry_has_required_keys
  - Inertia is positive for non-trivial data                    -> test_inertia_is_positive_for_nonempty_data
  - Labels are valid cluster indices in range [0, k)            -> test_labels_in_valid_range
  - centers dtype is float (not int)                            -> test_centers_are_float
  - Single-sample input raises ValueError                       -> test_single_sample_raises_value_error
  - N == max(k_values) exactly does not crash                   -> test_n_equals_max_k_boundary
  - Different random_state produces different labels            -> test_different_random_states_differ

Total: 17 tests (8 from ROADMAP, 9 additional)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# Pattern 8: import bootstrap — tests/ is one level below repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# These imports will fail with ImportError until the module is implemented.
# That is the expected initial failure mode.
from usv_spectrogram.classification.cluster_sweep import ClusterSweepConfig, run_sweep  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_blobs_3(n_per_cluster: int = 200, n_features: int = 10, seed: int = 0):
    """Return well-separated 3-cluster blob features and true labels."""
    X, y = make_blobs(
        n_samples=n_per_cluster * 3,
        n_features=n_features,
        centers=3,
        cluster_std=0.5,
        random_state=seed,
    )
    return X.astype(np.float32), y


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestClusterSweepConfig:
    def test_cluster_sweep_config_defaults(self):
        """Verify ClusterSweepConfig default fields match the spec exactly.

        Spec (ROADMAP 17.7): k_values=(5,7,10,15,20,27), random_state=42,
        n_init=10, silhouette_sample_size=2000.
        """
        cfg = ClusterSweepConfig()
        assert cfg.k_values == (5, 7, 10, 15, 20, 27), (
            f"Expected (5,7,10,15,20,27), got {cfg.k_values}"
        )
        assert cfg.random_state == 42
        assert cfg.n_init == 10
        assert cfg.silhouette_sample_size == 2000

    def test_cluster_sweep_config_is_frozen(self):
        """Verify ClusterSweepConfig is a frozen dataclass (Pattern 1).

        Mutating any field must raise FrozenInstanceError (or AttributeError),
        guaranteeing configs are not accidentally modified mid-pipeline.
        """
        cfg = ClusterSweepConfig()
        with pytest.raises((TypeError, AttributeError)):
            cfg.random_state = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ROADMAP test plan — 8 required tests
# ---------------------------------------------------------------------------

class TestRunSweepBlobData:
    """Tests that use synthetic make_blobs data."""

    @pytest.fixture(autouse=True)
    def blob_features(self):
        """600-point, 10-dimensional, 3-cluster blob dataset."""
        self.X, self.y = _make_blobs_3(n_per_cluster=200, n_features=10, seed=7)
        self.N, self.D = self.X.shape  # 600, 10

    def test_best_k_has_highest_silhouette_on_blob_data(self):
        """ROADMAP test 1: k=3 must have the highest silhouette on 3-blob data.

        When the data has exactly 3 well-separated Gaussian clusters, k-means
        with k=3 should yield a silhouette score strictly higher than k=2, 4,
        or 5 — because those mis-specify the number of clusters.
        We use k_values=(2,3,4,5) per spec note so k=3 is a candidate.
        """
        cfg = ClusterSweepConfig(k_values=(2, 3, 4, 5), random_state=42, n_init=10)
        results = run_sweep(self.X, cfg)

        silhouettes = {k: results[k]["silhouette"] for k in cfg.k_values}
        best_k = max(silhouettes, key=silhouettes.__getitem__)
        assert best_k == 3, (
            f"Expected k=3 to have highest silhouette on 3-blob data, "
            f"but got k={best_k}. Silhouettes: {silhouettes}"
        )

    def test_output_keys_match_k_values(self):
        """ROADMAP test 2: output dict must have exactly one key per k in cfg.k_values."""
        cfg = ClusterSweepConfig(k_values=(2, 3, 5), random_state=42, n_init=5)
        results = run_sweep(self.X, cfg)
        assert set(results.keys()) == set(cfg.k_values), (
            f"Output keys {set(results.keys())} != k_values {set(cfg.k_values)}"
        )

    def test_output_shapes_per_k(self):
        """ROADMAP test 3: labels shape=(N,) and centers shape=(k,D) for each k.

        The number of labels must equal N (one per sample). Centers must have
        exactly k rows and D columns (matching the feature dimensionality).
        """
        cfg = ClusterSweepConfig(k_values=(2, 3, 4), random_state=42, n_init=5)
        results = run_sweep(self.X, cfg)
        for k in cfg.k_values:
            labels = results[k]["labels"]
            centers = results[k]["centers"]
            assert labels.shape == (self.N,), (
                f"k={k}: labels.shape={labels.shape}, expected ({self.N},)"
            )
            assert centers.shape == (k, self.D), (
                f"k={k}: centers.shape={centers.shape}, expected ({k},{self.D})"
            )

    def test_inertia_monotonically_non_increasing(self):
        """ROADMAP test 4: inertia[k] >= inertia[k+1] for all consecutive k.

        Adding more clusters can only reduce (or preserve) the within-cluster
        sum of squares, so inertia must be monotonically non-increasing as k
        grows. This is a mathematical guarantee of k-means, not a heuristic.
        """
        cfg = ClusterSweepConfig(k_values=(2, 3, 5, 7, 10), random_state=42, n_init=10)
        results = run_sweep(self.X, cfg)
        sorted_ks = sorted(cfg.k_values)
        inertias = [results[k]["inertia"] for k in sorted_ks]
        for i in range(len(inertias) - 1):
            k_lo, k_hi = sorted_ks[i], sorted_ks[i + 1]
            assert inertias[i] >= inertias[i + 1], (
                f"Inertia not non-increasing: inertia[k={k_lo}]={inertias[i]:.4f} "
                f"< inertia[k={k_hi}]={inertias[i + 1]:.4f}"
            )

    def test_silhouette_subsample_close_to_full(self):
        """ROADMAP test 5: silhouette via subsample matches full silhouette within 0.05.

        The implementation subsamples at most silhouette_sample_size points for
        speed. On N=600 data this subsample may equal the full set; either way
        the returned value must agree with sklearn's ground-truth within 0.05.
        """
        # Use a large subsample so we reliably cover all 600 points
        cfg = ClusterSweepConfig(
            k_values=(3,), random_state=42, n_init=10, silhouette_sample_size=2000
        )
        results = run_sweep(self.X, cfg)
        reported = results[3]["silhouette"]

        labels = results[3]["labels"]
        full_silhouette = silhouette_score(self.X, labels, random_state=42)

        assert abs(reported - full_silhouette) <= 0.05, (
            f"Reported silhouette {reported:.4f} differs from full silhouette "
            f"{full_silhouette:.4f} by more than 0.05"
        )

    def test_same_random_state_produces_same_labels(self):
        """ROADMAP test 8: identical random_state must yield identical label arrays.

        Reproducibility is critical for downstream SIS comparisons — if the
        labeling changes between runs, benchmark comparisons become meaningless.
        """
        cfg = ClusterSweepConfig(k_values=(3, 5), random_state=42, n_init=10)
        results_a = run_sweep(self.X, cfg)
        results_b = run_sweep(self.X, cfg)
        for k in cfg.k_values:
            np.testing.assert_array_equal(
                results_a[k]["labels"],
                results_b[k]["labels"],
                err_msg=f"Labels differ between two runs at k={k} with same random_state",
            )


class TestRunSweepEdgeCases:
    """Tests for error handling and boundary conditions."""

    def test_empty_features_raises_value_error(self):
        """ROADMAP test 6: an empty feature matrix (0 rows) must raise ValueError.

        Passing an empty array to run_sweep is a programming error; the function
        must fail loudly rather than silently return empty results.
        """
        cfg = ClusterSweepConfig(k_values=(3, 5), random_state=42)
        empty_features = np.empty((0, 10), dtype=np.float32)
        with pytest.raises(ValueError):
            run_sweep(empty_features, cfg)

    def test_small_n_skips_large_k_with_warning(self):
        """ROADMAP test 7: when N < max(k_values), oversized k values are skipped with a warning.

        K-means requires N >= k. Rather than crashing, the implementation must
        emit a UserWarning for each skipped k and return only the feasible k
        entries in the results dict.
        """
        # 4 samples, 3 features; default k_values goes up to 27, only k=2,3
        # are feasible. We test that we get a warning and the dict only contains
        # k values where k <= N.
        rng = np.random.default_rng(0)
        small_X = rng.standard_normal((4, 3)).astype(np.float32)
        cfg = ClusterSweepConfig(k_values=(2, 3, 5), random_state=42, n_init=3)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = run_sweep(small_X, cfg)

        # k=5 must be skipped (N=4 < k=5)
        assert 5 not in results, (
            "k=5 should be skipped when N=4, but it appears in results"
        )
        # At least one warning must have been emitted about skipping
        warning_messages = [str(w.message) for w in caught]
        assert any("5" in msg or "skip" in msg.lower() or "k" in msg.lower()
                   for msg in warning_messages), (
            f"Expected a warning about skipped k=5, got: {warning_messages}"
        )
        # Feasible k values must still be present
        assert 2 in results
        assert 3 in results


# ---------------------------------------------------------------------------
# Additional gap-pattern tests
# ---------------------------------------------------------------------------

class TestRunSweepAdditional:
    """Additional correctness tests covering recurring gap patterns."""

    @pytest.fixture(autouse=True)
    def blob_features(self):
        self.X, self.y = _make_blobs_3(n_per_cluster=150, n_features=8, seed=13)
        self.N, self.D = self.X.shape  # 450, 8

    def test_each_result_entry_has_required_keys(self):
        """Each dict value in the results must contain the four required keys.

        Spec: {k: {'labels', 'inertia', 'silhouette', 'centers'}}. Missing
        any key would break downstream SIS/benchmark code that expects them.
        """
        cfg = ClusterSweepConfig(k_values=(3, 5), random_state=42, n_init=5)
        results = run_sweep(self.X, cfg)
        required_keys = {"labels", "inertia", "silhouette", "centers"}
        for k in cfg.k_values:
            assert required_keys.issubset(results[k].keys()), (
                f"k={k} result missing keys: {required_keys - results[k].keys()}"
            )

    def test_inertia_is_positive_for_nonempty_data(self):
        """Inertia must be strictly positive for any non-degenerate input.

        Zero inertia would imply every point is its own center, which only
        happens when k == N. Positive inertia is a sanity-check on the
        k-means objective value.
        """
        cfg = ClusterSweepConfig(k_values=(3,), random_state=42, n_init=5)
        results = run_sweep(self.X, cfg)
        assert results[3]["inertia"] > 0.0, (
            f"Expected positive inertia, got {results[3]['inertia']}"
        )

    def test_labels_in_valid_range(self):
        """All label values must lie in [0, k) for each k.

        An out-of-range label index would crash any downstream lookup into a
        centers array or confusion matrix.
        """
        cfg = ClusterSweepConfig(k_values=(3, 5, 7), random_state=42, n_init=5)
        results = run_sweep(self.X, cfg)
        for k in cfg.k_values:
            labels = results[k]["labels"]
            assert labels.min() >= 0, f"k={k}: negative label {labels.min()}"
            assert labels.max() < k, (
                f"k={k}: label {labels.max()} out of range [0, {k})"
            )

    def test_centers_are_float(self):
        """Centers array must have a floating-point dtype.

        Integer centers would corrupt distance computations in downstream code
        that computes Euclidean distances without explicit casting.
        """
        cfg = ClusterSweepConfig(k_values=(3,), random_state=42, n_init=5)
        results = run_sweep(self.X, cfg)
        centers = results[3]["centers"]
        assert np.issubdtype(centers.dtype, np.floating), (
            f"Expected float centers, got dtype={centers.dtype}"
        )

    def test_single_sample_raises_value_error(self):
        """A single-sample input must raise ValueError (k > N=1 is always infeasible).

        This boundary check ensures the function does not silently pass by
        returning empty or partially-filled results for degenerate input.
        """
        single = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        cfg = ClusterSweepConfig(k_values=(2, 3), random_state=42)
        with pytest.raises((ValueError, UserWarning)):
            # If all k values are skipped due to N<k, either a ValueError or a
            # situation where no results are returned is acceptable; the key
            # requirement is no silent data corruption.
            results = run_sweep(single, cfg)
            # If it doesn't raise, then no feasible k must have been run
            assert len(results) == 0, (
                "Single-sample input should yield no results or raise"
            )

    def test_n_equals_max_k_boundary(self):
        """When N exactly equals max(k_values), that k must succeed without error.

        This is the boundary condition where every sample becomes its own
        cluster center. It should not crash even though it's degenerate.
        """
        k_max = 5
        # N == k_max == 5, features have 3 dimensions, each point is unique
        rng = np.random.default_rng(99)
        X_boundary = rng.standard_normal((k_max, 3)).astype(np.float32)
        cfg = ClusterSweepConfig(k_values=(2, k_max), random_state=42, n_init=3)
        # Should not raise; k=k_max is at the boundary but still feasible
        results = run_sweep(X_boundary, cfg)
        assert k_max in results, (
            f"k={k_max} should be present when N={k_max}"
        )

    def test_different_random_states_differ(self):
        """Two runs with different random_state values must (with high probability) differ.

        This is the counterpart of the reproducibility test — it confirms that
        random_state actually controls randomness rather than being ignored.
        We use data that is genuinely ambiguous (overlapping clusters) so that
        initialisation differences propagate to different final assignments.
        """
        rng_data = np.random.default_rng(42)
        # Overlapping clusters: high within-cluster std relative to separation
        X_ambig, _ = make_blobs(
            n_samples=300, n_features=5, centers=5,
            cluster_std=3.0, random_state=0,
        )
        X_ambig = X_ambig.astype(np.float32)

        cfg_a = ClusterSweepConfig(k_values=(7,), random_state=0, n_init=1)
        cfg_b = ClusterSweepConfig(k_values=(7,), random_state=999, n_init=1)

        results_a = run_sweep(X_ambig, cfg_a)
        results_b = run_sweep(X_ambig, cfg_b)

        # With different seeds and n_init=1 the assignments should differ
        labels_a = results_a[7]["labels"]
        labels_b = results_b[7]["labels"]
        # Allow a small number of matches by chance, but the majority should differ
        match_fraction = np.mean(labels_a == labels_b)
        assert match_fraction < 0.95, (
            f"Labels are suspiciously identical across different random states "
            f"(match fraction={match_fraction:.2f}). random_state may be ignored."
        )
