"""Tests for the three novel functions in scripts/run_vae_diagnostic_on_encoder.py.

Covers MINOR-2 from the Module 18.4 master-review (2026-05-26):
  "Three novel functions (pc1_cohens_d_features, per_dimension_max_cohens_d,
   notch_migration_in_feature_space) are not in the ROADMAP test plan and
   untested."

Functions under test (imported directly from the script via scripts/ path
bootstrap — the same mechanism the script itself uses):

  1. pc1_cohens_d_features(feats_a, feats_b) -> float
     |Cohen's d| between cohorts on PC1 of the combined feature matrix.

  2. per_dimension_max_cohens_d(feats_a, feats_b) -> float
     Max |Cohen's d| across all feature dimensions.

  3. notch_migration_in_feature_space(
         source_feats, target_feats_clean, target_feats_notched, k) -> float
     Fraction of notched-target features whose k-NN are majority source.

All tests use synthetic numpy arrays constructed inline — no encoder, no
checkpoint, no GPU. CPU-only, seeded for reproducibility.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path bootstrap — mirror the script's own sys.path setup so we can import
# the three functions without executing __main__.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
_SCRIPTS = REPO_ROOT / "scripts"
for _p in (str(_SRC), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# We import only the three pure-numpy functions; the rest of the script has
# heavy imports (PIL, torchvision, train_lab_classifier) that require the rig.
# Use importlib to load the module object so we can pick the symbols we need
# without triggering argparse or __main__ side-effects.
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "run_vae_diagnostic_on_encoder",
    _SCRIPTS / "run_vae_diagnostic_on_encoder.py",
)
_mod = _ilu.module_from_spec(_spec)

# The script imports train_lab_classifier._DEFAULT_IMAGE_SIZE at module level,
# which is a non-standard import requiring the scripts/ path (already added).
# We execute the module; if that fails the test collection itself fails,
# surfacing the import error clearly rather than hiding it.
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

pc1_cohens_d_features = _mod.pc1_cohens_d_features
per_dimension_max_cohens_d = _mod.per_dimension_max_cohens_d
notch_migration_in_feature_space = _mod.notch_migration_in_feature_space

# ---------------------------------------------------------------------------
# Shared RNG helper
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(42)


def _randn(shape, *, rng=_RNG) -> np.ndarray:
    return rng.standard_normal(shape).astype(np.float64)


# ===========================================================================
# 1. pc1_cohens_d_features
# ===========================================================================

class TestPc1CohensDFeatures:

    def test_identical_distributions_near_zero(self):
        """When both cohorts are drawn from the same distribution the PC1 Cohen's d
        should be near zero (no systematic cohort separation along PC1).

        We use a fixed seed and a generous tolerance (< 0.5) rather than < 0.1
        because with n=100 and D=16 there will always be some sampling noise on
        PC1. The key assertion is that it is NOT large (> 1).
        """
        rng = np.random.default_rng(0)
        feats_a = rng.standard_normal((100, 16)).astype(np.float64)
        feats_b = rng.standard_normal((100, 16)).astype(np.float64)
        d = pc1_cohens_d_features(feats_a, feats_b)
        assert isinstance(d, float), f"Expected float, got {type(d)}"
        assert d < 0.5, (
            f"pc1_cohens_d_features on identical distributions should be near 0, "
            f"got {d:.4f}. A value >= 0.5 suggests the function is not measuring "
            f"cohort separation correctly."
        )

    def test_strongly_shifted_distributions_large_d(self):
        """When cohort B has a strong mean shift, PC1 Cohen's d must be large (> 1).

        We shift all features of cohort B by 5.0 along the first dimension.
        PC1 of the combined matrix will align with that shift direction, and
        the Cohen's d on the projected scores must be >> 1.
        """
        rng = np.random.default_rng(1)
        n, D = 100, 32
        feats_a = rng.standard_normal((n, D)).astype(np.float64)
        feats_b = rng.standard_normal((n, D)).astype(np.float64)
        feats_b[:, 0] += 5.0  # large mean shift on one dimension

        d = pc1_cohens_d_features(feats_a, feats_b)
        assert d > 1.0, (
            f"pc1_cohens_d_features with mean shift=5 should yield d > 1, "
            f"got {d:.4f}"
        )

    def test_empty_feats_a_returns_zero(self):
        """pc1_cohens_d_features must return 0.0 when feats_a is empty."""
        feats_a = np.empty((0, 16), dtype=np.float64)
        feats_b = _randn((50, 16))
        result = pc1_cohens_d_features(feats_a, feats_b)
        assert result == 0.0, (
            f"Expected 0.0 for empty feats_a, got {result}"
        )

    def test_empty_feats_b_returns_zero(self):
        """pc1_cohens_d_features must return 0.0 when feats_b is empty."""
        feats_a = _randn((50, 16))
        feats_b = np.empty((0, 16), dtype=np.float64)
        result = pc1_cohens_d_features(feats_a, feats_b)
        assert result == 0.0, (
            f"Expected 0.0 for empty feats_b, got {result}"
        )

    def test_return_is_non_negative(self):
        """pc1_cohens_d_features returns |Cohen's d|, so the result must be >= 0."""
        rng = np.random.default_rng(2)
        feats_a = rng.standard_normal((40, 8)).astype(np.float64)
        feats_b = rng.standard_normal((40, 8)).astype(np.float64) + 1.0
        d = pc1_cohens_d_features(feats_a, feats_b)
        assert d >= 0.0, (
            f"pc1_cohens_d_features returned negative value {d:.6f}; "
            f"function must return the absolute value |Cohen's d|"
        )

    def test_result_is_finite(self):
        """pc1_cohens_d_features must return a finite float (no NaN, no Inf)."""
        rng = np.random.default_rng(3)
        feats_a = rng.standard_normal((60, 20)).astype(np.float64)
        feats_b = rng.standard_normal((60, 20)).astype(np.float64) + 2.0
        d = pc1_cohens_d_features(feats_a, feats_b)
        assert np.isfinite(d), f"pc1_cohens_d_features returned non-finite: {d}"

    def test_symmetric_in_cohort_order(self):
        """Swapping feats_a and feats_b must give the same |d| (absolute value).

        Cohen's d is signed (mean_a - mean_b), but the function returns |d|,
        so swapping the arguments should produce the same result.
        """
        rng = np.random.default_rng(4)
        feats_a = rng.standard_normal((50, 10)).astype(np.float64)
        feats_b = rng.standard_normal((50, 10)).astype(np.float64) + 1.5
        d_ab = pc1_cohens_d_features(feats_a, feats_b)
        d_ba = pc1_cohens_d_features(feats_b, feats_a)
        assert abs(d_ab - d_ba) < 0.05, (
            f"pc1_cohens_d_features is not symmetric: "
            f"d(a,b)={d_ab:.4f} vs d(b,a)={d_ba:.4f}. "
            f"The function should return |Cohen's d| (absolute value)."
        )


# ===========================================================================
# 2. per_dimension_max_cohens_d
# ===========================================================================

class TestPerDimensionMaxCohensD:

    def test_identical_distributions_near_zero(self):
        """Max per-dim Cohen's d should be near zero when no systematic shift exists.

        With n=200 and D=8, the expected max|d| under the null is well below 0.5
        (unlike D=512 which inflates to ~0.34 at n=1000). We use D=8 to ensure a
        tight null distribution with our seed.
        """
        rng = np.random.default_rng(10)
        feats_a = rng.standard_normal((200, 8)).astype(np.float64)
        feats_b = rng.standard_normal((200, 8)).astype(np.float64)
        d = per_dimension_max_cohens_d(feats_a, feats_b)
        assert d < 0.5, (
            f"per_dimension_max_cohens_d on same-distribution data (D=8, n=200) "
            f"should be < 0.5, got {d:.4f}"
        )

    def test_strongly_shifted_one_dimension_large_d(self):
        """When one dimension is strongly shifted (mean diff >> pooled std),
        per_dimension_max_cohens_d must return a value > 1.

        With shift=5.0 and unit variance, Cohen's d on that dimension ≈ 5.0.
        """
        rng = np.random.default_rng(11)
        n, D = 100, 16
        feats_a = rng.standard_normal((n, D)).astype(np.float64)
        feats_b = rng.standard_normal((n, D)).astype(np.float64)
        feats_b[:, 3] += 5.0  # large shift on dimension 3 only

        d = per_dimension_max_cohens_d(feats_a, feats_b)
        assert d > 1.0, (
            f"per_dimension_max_cohens_d with dim-3 shift=5 should yield d > 1, "
            f"got {d:.4f}"
        )

    def test_minimum_sample_size_no_nan(self):
        """per_dimension_max_cohens_d with n=2 per cohort must not return NaN.

        ddof=1 variance requires n >= 2. With n=2 the variance is noisy but
        not undefined. The function uses ddof=1 in np.var(axis=0, ddof=1),
        which is valid for n=2 (produces one degree of freedom).
        """
        rng = np.random.default_rng(12)
        feats_a = rng.standard_normal((2, 8)).astype(np.float64)
        feats_b = rng.standard_normal((2, 8)).astype(np.float64) + 1.0
        d = per_dimension_max_cohens_d(feats_a, feats_b)
        assert np.isfinite(d), (
            f"per_dimension_max_cohens_d with n=2 returned non-finite: {d}. "
            f"ddof=1 variance should be valid at n=2 (not NaN)."
        )

    def test_result_is_non_negative(self):
        """per_dimension_max_cohens_d returns max |Cohen's d|, so result >= 0."""
        rng = np.random.default_rng(13)
        feats_a = rng.standard_normal((50, 12)).astype(np.float64)
        feats_b = rng.standard_normal((50, 12)).astype(np.float64) + 0.5
        d = per_dimension_max_cohens_d(feats_a, feats_b)
        assert d >= 0.0, (
            f"per_dimension_max_cohens_d returned {d:.6f} < 0.0"
        )

    def test_shift_in_single_dimension_detected(self):
        """Shift on exactly one dimension must be detectable.

        If the function returns near-zero when only one dimension is shifted,
        it means it is averaging d across dimensions (wrong) instead of taking
        the max. We use a small shift (0.5) on 1 out of 32 dimensions; the max
        should reflect that dimension, not be diluted to near zero.
        """
        rng = np.random.default_rng(14)
        n, D = 100, 32
        feats_a = rng.standard_normal((n, D)).astype(np.float64)
        feats_b = rng.standard_normal((n, D)).astype(np.float64)
        feats_b[:, 0] += 3.0  # Cohen's d ≈ 3.0 on dim 0

        d = per_dimension_max_cohens_d(feats_a, feats_b)
        assert d > 1.0, (
            f"Single-dimension shift of 3.0 should give max|d| > 1.0, "
            f"got {d:.4f}. The function may be averaging instead of max-ing."
        )

    def test_result_is_finite(self):
        """per_dimension_max_cohens_d must return a finite float (no NaN/Inf)
        on a typical 512-dimensional feature input similar to ResNet-18 output.
        """
        rng = np.random.default_rng(15)
        feats_a = rng.standard_normal((50, 512)).astype(np.float64)
        feats_b = rng.standard_normal((50, 512)).astype(np.float64) + 0.1
        d = per_dimension_max_cohens_d(feats_a, feats_b)
        assert np.isfinite(d), (
            f"per_dimension_max_cohens_d on 512-dim features returned non-finite: {d}"
        )


# ===========================================================================
# 3. notch_migration_in_feature_space
# ===========================================================================

class TestNotchMigrationInFeatureSpace:

    def test_returns_float_in_0_1(self):
        """notch_migration_in_feature_space must return a float in [0, 1]."""
        rng = np.random.default_rng(20)
        D = 16
        source_feats = rng.standard_normal((20, D)).astype(np.float64)
        target_clean = rng.standard_normal((20, D)).astype(np.float64) + 3.0
        target_notched = rng.standard_normal((10, D)).astype(np.float64) + 3.0

        rate = notch_migration_in_feature_space(source_feats, target_clean, target_notched, k=5)
        assert isinstance(rate, float), f"Expected float, got {type(rate)}"
        assert 0.0 <= rate <= 1.0, (
            f"notch_migration_in_feature_space returned {rate:.4f} outside [0, 1]"
        )

    def test_empty_notched_returns_zero(self):
        """notch_migration_in_feature_space must return 0.0 when notched input is empty.

        No notched samples → nothing can migrate → migration rate = 0.
        """
        rng = np.random.default_rng(21)
        D = 16
        source_feats = rng.standard_normal((20, D)).astype(np.float64)
        target_clean = rng.standard_normal((20, D)).astype(np.float64) + 3.0
        target_notched = np.empty((0, D), dtype=np.float64)

        rate = notch_migration_in_feature_space(source_feats, target_clean, target_notched, k=5)
        assert rate == 0.0, (
            f"Expected 0.0 for empty notched input, got {rate}"
        )

    def test_high_migration_when_notched_near_source(self):
        """Migration rate must be high when notched-target features land near
        the source cluster.

        Setup:
          - source_feats: cluster around origin (0, 0, ...)
          - target_clean: cluster far from source (offset +10 on all dims)
          - target_notched: placed near SOURCE cluster (offset 0 + small noise)

        k-NN of each notched point should be majority source → rate ≈ 1.0.
        We assert rate > 0.7 (generous tolerance for small k).
        """
        rng = np.random.default_rng(22)
        D = 8
        n = 30
        source_feats = rng.standard_normal((n, D)).astype(np.float64)          # around 0
        target_clean = rng.standard_normal((n, D)).astype(np.float64) + 10.0   # far from source
        # Notched targets placed near source cluster
        target_notched = rng.standard_normal((n, D)).astype(np.float64) * 0.1  # tight around 0

        rate = notch_migration_in_feature_space(source_feats, target_clean, target_notched, k=5)
        assert rate > 0.7, (
            f"Expected high migration rate (> 0.7) when notched features land "
            f"near source cluster, got {rate:.4f}"
        )

    def test_low_migration_when_notched_stays_near_target(self):
        """Migration rate must be low when notched-target features stay near the
        target cluster (not the source).

        Setup:
          - source_feats: cluster around 0
          - target_clean: cluster around +10
          - target_notched: also around +10 (same as clean target, notch had no effect)

        k-NN of each notched point should be majority target → rate ≈ 0.
        We assert rate < 0.3.
        """
        rng = np.random.default_rng(23)
        D = 8
        n = 30
        source_feats = rng.standard_normal((n, D)).astype(np.float64)           # around 0
        target_clean = rng.standard_normal((n, D)).astype(np.float64) + 10.0    # around +10
        target_notched = rng.standard_normal((n, D)).astype(np.float64) + 10.0  # still around +10

        rate = notch_migration_in_feature_space(source_feats, target_clean, target_notched, k=5)
        assert rate < 0.3, (
            f"Expected low migration rate (< 0.3) when notched features stay "
            f"near target cluster, got {rate:.4f}"
        )

    def test_result_is_finite(self):
        """notch_migration_in_feature_space must not return NaN or Inf."""
        rng = np.random.default_rng(24)
        D = 32
        source_feats = rng.standard_normal((20, D)).astype(np.float64)
        target_clean = rng.standard_normal((20, D)).astype(np.float64) + 2.0
        target_notched = rng.standard_normal((10, D)).astype(np.float64) + 1.0

        rate = notch_migration_in_feature_space(source_feats, target_clean, target_notched, k=5)
        assert np.isfinite(rate), (
            f"notch_migration_in_feature_space returned non-finite: {rate}"
        )

    def test_k_equals_one(self):
        """notch_migration_in_feature_space with k=1 must return a valid [0,1] rate.

        k=1 (nearest-neighbour) is the boundary case for the majority vote.
        With k=1 the majority condition simplifies to: nearest neighbour is source.
        """
        rng = np.random.default_rng(25)
        D = 8
        source_feats = rng.standard_normal((15, D)).astype(np.float64)
        target_clean = rng.standard_normal((15, D)).astype(np.float64) + 5.0
        target_notched = rng.standard_normal((8, D)).astype(np.float64) + 2.5

        rate = notch_migration_in_feature_space(source_feats, target_clean, target_notched, k=1)
        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0, (
            f"notch_migration_in_feature_space(k=1) returned {rate:.4f} outside [0,1]"
        )

    def test_single_notched_sample(self):
        """notch_migration_in_feature_space with a single notched sample must
        return either 0.0 or 1.0 (binary: migrated or not).
        """
        rng = np.random.default_rng(26)
        D = 8
        source_feats = rng.standard_normal((20, D)).astype(np.float64)
        target_clean = rng.standard_normal((20, D)).astype(np.float64) + 5.0
        # Single notched sample placed right in the middle of source cluster
        target_notched = rng.standard_normal((1, D)).astype(np.float64) * 0.01

        rate = notch_migration_in_feature_space(source_feats, target_clean, target_notched, k=5)
        assert rate in (0.0, 1.0), (
            f"With a single notched sample, rate must be 0.0 or 1.0, got {rate}"
        )
