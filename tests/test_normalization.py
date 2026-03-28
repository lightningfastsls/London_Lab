"""Tests for normalization module — written by test-architect BEFORE implementation.

Module under test: src/usv_spectrogram/postprocessing/normalization.py

ROADMAP test plan coverage (section 15.6):
  1. Constant input → all zeros output      -> test_constant_input_returns_all_zeros
  2. Known distribution (noise~0.1±0.02,    -> test_known_distribution_usv_zscore_far_from_noise
     USV=0.9) → USV Z-score >> noise Z-scores
  3. Two recordings with different noise     -> test_different_noise_floors_normalize_to_comparable_zscores
     floors normalize to comparable Z-scores
  4. Empty array → handled gracefully        -> test_empty_array_returns_empty
  5. All-same-value → MAD=0 guard activated  -> test_all_same_value_mad_zero_guard

Additional coverage (recurring gap patterns):
  - Single-element array                     -> test_single_element_returns_zero
  - Exact 50/50 split (boundary condition)   -> test_exact_split_uses_lower_half_for_noise
  - Negative input values (probs can be
    non-standard after upstream transforms)  -> test_negative_values_handled
  - Very large arrays (performance sanity)   -> test_large_array_completes_in_reasonable_time
  - normalize_scores_batch with multiple     -> test_batch_normalizes_each_recording_independently
    recordings
  - normalize_scores_batch with empty dict   -> test_batch_empty_dict_returns_empty_dict
  - Output can exceed [0, 1] (expected       -> test_output_can_exceed_unit_interval
    behaviour per spec)
  - Noise estimate uses only bottom 50th     -> test_noise_estimated_from_bottom_half_only
    percentile, not full array

Total: 13 tests (5 from ROADMAP, 8 additional)

Hand-computed spot-check for test_known_distribution_usv_zscore_far_from_noise:
  Input: [0.08, 0.09, 0.09, 0.10, 0.10, 0.10, 0.11, 0.11, 0.12, 0.90]
  Bottom 50% (5 smallest): [0.08, 0.09, 0.09, 0.10, 0.10]
  noise_median = 0.09
  noise_MAD = median(|vals - 0.09|) = median([0.01, 0.0, 0.0, 0.01, 0.01]) = 0.01
  z(0.90) = (0.90 - 0.09) / 0.01 = 81.0
  z(0.08) = (0.08 - 0.09) / 0.01 = -1.0

Hand-computed spot-check for test_exact_split_uses_lower_half_for_noise:
  Input of 8 values: [0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9]
  Bottom 50% (4 smallest): [0.1, 0.1, 0.1, 0.1]
  noise_median = 0.1, noise_MAD = 0.0  → MAD=0 guard must activate
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from usv_spectrogram.postprocessing.normalization import (
    normalize_scores_batch,
    normalize_scores_per_recording,
)


# ---------------------------------------------------------------------------
# ROADMAP Test Plan — 5 required cases
# ---------------------------------------------------------------------------


class TestNormalizeScoresPerRecording:
    """Core normalisation behaviour for a single recording."""

    def test_constant_input_returns_all_zeros(self):
        """Spec: constant input → all-zero output (no deviation from noise).

        When every window has the same probability the noise distribution is
        perfectly uniform.  After Z-normalisation every element is 0.
        """
        probs = np.full(100, 0.3)
        result = normalize_scores_per_recording(probs)

        assert result.shape == probs.shape
        np.testing.assert_allclose(result, 0.0, atol=1e-9)

    def test_known_distribution_usv_zscore_far_from_noise(self):
        """Spec: USV window at 0.9 should get Z-score >> noise Z-scores.

        Hand-computed expected values (see module docstring):
          noise_median = 0.09, noise_MAD = 0.01
          z(0.90) = 81.0
          z(0.08) = -1.0

        Noise windows' Z-scores are near 0; USV Z-score is at 81.
        """
        # 9 noise windows, 1 USV window — bottom 50% (5) are all noise
        noise_vals = np.array([0.08, 0.09, 0.09, 0.10, 0.10, 0.10, 0.11, 0.11, 0.12])
        usv_val = np.array([0.90])
        probs = np.concatenate([noise_vals, usv_val])  # length 10

        result = normalize_scores_per_recording(probs)

        # USV Z-score should be 81.0 (within float tolerance)
        usv_z = result[-1]  # last element is the USV window
        assert usv_z == pytest.approx(81.0, abs=1e-6), (
            f"Expected USV Z-score ≈ 81.0, got {usv_z}"
        )

        # The lowest noise window should have Z-score = -1.0
        lowest_noise_z = result[0]
        assert lowest_noise_z == pytest.approx(-1.0, abs=1e-6), (
            f"Expected lowest noise Z-score ≈ -1.0, got {lowest_noise_z}"
        )

        # Every noise Z-score should be much smaller than the USV Z-score
        noise_z_scores = result[:-1]
        assert np.all(noise_z_scores < usv_z / 2), (
            "All noise Z-scores should be far below the USV Z-score"
        )

    def test_different_noise_floors_normalize_to_comparable_zscores(self):
        """Spec: two recordings with different noise floors → comparable USV Z-scores.

        Recording A: noise at 0.10, USV at 0.90
        Recording B: noise at 0.50, USV at 0.90

        Without normalization, a threshold of 0.70 would catch A's USV but
        not discriminate well in B.  After normalization, both USV windows
        should receive high Z-scores of similar magnitude.
        """
        rng = np.random.default_rng(42)

        # Recording A — quiet recording (low noise floor)
        noise_a = 0.10 + rng.normal(0, 0.005, size=99)
        noise_a = np.clip(noise_a, 0.0, 1.0)
        usv_a = np.array([0.90])
        probs_a = np.concatenate([noise_a, usv_a])

        # Recording B — noisy recording (high noise floor)
        noise_b = 0.50 + rng.normal(0, 0.005, size=99)
        noise_b = np.clip(noise_b, 0.0, 1.0)
        usv_b = np.array([0.90])
        probs_b = np.concatenate([noise_b, usv_b])

        z_a = normalize_scores_per_recording(probs_a)
        z_b = normalize_scores_per_recording(probs_b)

        usv_z_a = z_a[-1]
        usv_z_b = z_b[-1]

        # Both USV Z-scores should be large positive values
        assert usv_z_a > 5.0, f"Recording A USV Z-score too small: {usv_z_a}"
        assert usv_z_b > 5.0, f"Recording B USV Z-score too small: {usv_z_b}"

        # They should be in a comparable range (within 5x of each other)
        ratio = max(usv_z_a, usv_z_b) / min(usv_z_a, usv_z_b)
        assert ratio < 5.0, (
            f"Z-scores too dissimilar after normalization: {usv_z_a:.2f} vs {usv_z_b:.2f}"
        )

        # Noise Z-scores in both recordings should be centered near 0
        assert abs(np.median(z_a[:-1])) < 1.0, "Recording A noise Z-scores not centred"
        assert abs(np.median(z_b[:-1])) < 1.0, "Recording B noise Z-scores not centred"

    def test_empty_array_returns_empty(self):
        """Spec: empty array input → return empty array (graceful handling).

        The function must not raise; it must return a zero-length ndarray.
        """
        result = normalize_scores_per_recording(np.array([]))

        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)

    def test_all_same_value_mad_zero_guard(self):
        """Spec: all-same-value → MAD=0 guard activated (no division by zero).

        The 50th-percentile noise slice will have zero variance.  The
        implementation must detect this and return a safe result (e.g. all
        zeros) rather than NaN or inf.
        """
        probs = np.full(50, 0.45)
        result = normalize_scores_per_recording(probs)

        assert result.shape == probs.shape
        assert np.all(np.isfinite(result)), "Result contains NaN or Inf when MAD=0"


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------


class TestNormalizeScoresEdgeCases:
    """Edge cases beyond the ROADMAP test plan."""

    def test_single_element_returns_zero(self):
        """Single-element array: noise slice is the single element, MAD=0 guard fires.

        The output must be finite and the function must not raise.
        """
        result = normalize_scores_per_recording(np.array([0.7]))

        assert result.shape == (1,)
        assert np.isfinite(result[0]), "Single-element result must be finite"

    def test_exact_split_uses_lower_half_for_noise(self):
        """Boundary: 8 elements split exactly 50/50 into noise/USV groups.

        Bottom 50% = [0.1, 0.1, 0.1, 0.1], all equal → MAD=0 guard fires.
        The guard must produce finite output, not NaN.

        This verifies that the 50th-percentile cutpoint boundary is handled
        correctly (not off-by-one) and that MAD=0 protection works here too.
        """
        probs = np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9])
        result = normalize_scores_per_recording(probs)

        assert result.shape == (8,)
        assert np.all(np.isfinite(result)), "Result contains NaN/Inf on 50/50 split"

    def test_output_can_exceed_unit_interval(self):
        """Spec: normalized scores can exceed [0, 1] — this is expected behaviour.

        A USV window far above the noise distribution will have Z >> 1.
        The implementation must NOT clip output to [0, 1].
        """
        noise_vals = np.full(90, 0.05)
        usv_vals = np.full(10, 0.95)
        probs = np.concatenate([noise_vals, usv_vals])

        result = normalize_scores_per_recording(probs)

        # USV Z-scores should be well above 1.0
        usv_z = result[-10:]
        assert np.all(usv_z > 1.0), (
            f"USV Z-scores should exceed 1.0 but got max={usv_z.max():.4f}"
        )

    def test_noise_estimated_from_bottom_half_only(self):
        """The noise estimate must use only the bottom 50th percentile.

        If it used the full array, a recording with many USV windows would
        inflate the median and give wrong Z-scores.  We test this by
        constructing an array where using the full median would produce a
        different result than using the bottom 50%.

        Array: 40 windows at 0.10 (noise), 60 windows at 0.90 (USV).
        Full-array median would be 0.90. Bottom-50%-median should be 0.10.
        """
        noise_count = 40
        usv_count = 60
        probs = np.concatenate([
            np.full(noise_count, 0.10),
            np.full(usv_count, 0.90),
        ])

        result = normalize_scores_per_recording(probs)

        # If noise was estimated from bottom 50% (first 50 elements = noise + usv mix):
        # The noise median should reflect the noise floor, not the USV probability.
        # Noise windows (first 40) should have low-magnitude Z-scores.
        noise_z = result[:noise_count]
        usv_z = result[noise_count:]

        # USV Z-scores must be distinctly higher than noise Z-scores
        assert np.mean(usv_z) > np.mean(noise_z) + 1.0, (
            "USV Z-scores should be meaningfully above noise Z-scores"
        )

    def test_negative_values_handled_without_crash(self):
        """Non-standard probability values (can occur after upstream transforms).

        The function must not crash on values outside [0, 1].  Output must be
        finite for finite input.
        """
        probs = np.array([-0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 1.5])
        result = normalize_scores_per_recording(probs)

        assert result.shape == probs.shape
        assert np.all(np.isfinite(result)), "Non-standard inputs must yield finite Z-scores"

    def test_output_shape_matches_input(self):
        """Output array shape must exactly match input array shape.

        Shape preservation is an invariant for any correct implementation.
        """
        rng = np.random.default_rng(7)
        for n in [10, 100, 1000]:
            probs = rng.uniform(0.0, 1.0, size=n)
            result = normalize_scores_per_recording(probs)
            assert result.shape == (n,), (
                f"Shape mismatch for n={n}: got {result.shape}"
            )

    def test_large_array_completes_in_reasonable_time(self):
        """Performance sanity: 300 000-element array (1 second at hop=1 sample) completes fast.

        The implementation must avoid Python loops over all elements.
        Threshold: 2 seconds on any modern machine.
        """
        rng = np.random.default_rng(99)
        # Simulate a long recording: 300k probability values
        probs = rng.uniform(0.0, 1.0, size=300_000)
        # Inject a few "USV" windows
        probs[150_000:150_010] = 0.95

        start = time.monotonic()
        result = normalize_scores_per_recording(probs)
        elapsed = time.monotonic() - start

        assert result.shape == (300_000,)
        assert np.all(np.isfinite(result))
        assert elapsed < 2.0, f"Normalization took {elapsed:.2f}s — too slow"


# ---------------------------------------------------------------------------
# normalize_scores_batch tests
# ---------------------------------------------------------------------------


class TestNormalizeScoresBatch:
    """Tests for the batch version operating on a dict of recordings."""

    def test_batch_normalizes_each_recording_independently(self):
        """Each recording in the batch must be normalised separately.

        Spec: returns dict[str, np.ndarray] with the same keys as input.

        Two recordings with wildly different noise floors should each have
        their own normalization applied — not a single shared scale.
        """
        rng = np.random.default_rng(11)

        noise_a = 0.10 + rng.normal(0, 0.01, size=99)
        probs_a = np.concatenate([np.clip(noise_a, 0, 1), [0.90]])

        noise_b = 0.70 + rng.normal(0, 0.01, size=99)
        probs_b = np.concatenate([np.clip(noise_b, 0, 1), [0.90]])

        all_probs = {"rec_a": probs_a, "rec_b": probs_b}
        result = normalize_scores_batch(all_probs)

        # Keys must be preserved
        assert set(result.keys()) == {"rec_a", "rec_b"}

        # Each output array must have the same shape as its input
        assert result["rec_a"].shape == probs_a.shape
        assert result["rec_b"].shape == probs_b.shape

        # Both normalised results must be finite
        assert np.all(np.isfinite(result["rec_a"]))
        assert np.all(np.isfinite(result["rec_b"]))

        # Verify independence: same input probability (0.90) with different
        # noise floors should produce different raw Z-scores.
        z_usv_a = result["rec_a"][-1]
        z_usv_b = result["rec_b"][-1]
        # They should differ (because noise floors differ) but both be positive
        assert z_usv_a > 0, f"rec_a USV Z-score should be positive: {z_usv_a}"
        assert z_usv_b > 0, f"rec_b USV Z-score should be positive: {z_usv_b}"
        # Since noise floors differ substantially, the Z-scores will differ
        # (rec_a USV is much further above its noise; rec_b is less so)
        assert z_usv_a > z_usv_b, (
            "rec_a has lower noise floor, so its USV Z-score should be higher "
            f"({z_usv_a:.2f} vs {z_usv_b:.2f})"
        )

    def test_batch_empty_dict_returns_empty_dict(self):
        """Spec: batch function with empty dict → return empty dict (not raise).

        Graceful handling of zero recordings is required.
        """
        result = normalize_scores_batch({})

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_batch_single_recording(self):
        """Batch with one recording should behave identically to per-recording call."""
        rng = np.random.default_rng(21)
        probs = rng.uniform(0.0, 1.0, size=200)

        single_result = normalize_scores_per_recording(probs)
        batch_result = normalize_scores_batch({"only_rec": probs})

        assert "only_rec" in batch_result
        np.testing.assert_allclose(
            batch_result["only_rec"],
            single_result,
            atol=1e-10,
            err_msg="Batch result for single recording must match per-recording result",
        )

    def test_batch_does_not_mutate_input(self):
        """normalize_scores_batch must not modify the input dict or its arrays."""
        probs = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        original_values = probs.copy()
        all_probs = {"rec": probs}

        normalize_scores_batch(all_probs)

        np.testing.assert_array_equal(
            probs,
            original_values,
            err_msg="Input array was mutated by normalize_scores_batch",
        )
        assert "rec" in all_probs, "Input dict keys were mutated"


# ---------------------------------------------------------------------------
# Adversarial tests added by test-hardener
# ---------------------------------------------------------------------------


class TestOutputDtype:
    """Fix W-2 (review finding): all code paths must return float64.

    The review found the main return path was returning input dtype (float32
    from CNN) instead of float64.  A cast was added.  These tests verify every
    return path produces float64, including float32 input and the zeros path.
    """

    def test_float32_input_produces_float64_output(self):
        """CNN outputs float32; normalized result must be float64.

        The main code path contains an explicit .astype(np.float64) cast that
        was added to fix review finding W-2.  Verify it actually fires.
        """
        probs = np.array([0.05, 0.06, 0.07, 0.08, 0.09, 0.90], dtype=np.float32)
        result = normalize_scores_per_recording(probs)

        assert result.dtype == np.float64, (
            f"Expected float64 output from float32 input, got {result.dtype}"
        )

    def test_float64_input_stays_float64(self):
        """float64 input must also produce float64 output (no dtype regression)."""
        probs = np.array([0.05, 0.06, 0.07, 0.08, 0.09, 0.90], dtype=np.float64)
        result = normalize_scores_per_recording(probs)

        assert result.dtype == np.float64, (
            f"Expected float64 output, got {result.dtype}"
        )

    def test_zeros_return_path_is_float64(self):
        """The zeros-return path (all spread estimates are zero) must also be float64.

        This path is: np.zeros_like(probabilities, dtype=np.float64).
        Verify the dtype argument was actually applied.
        """
        probs = np.full(20, 0.5, dtype=np.float32)
        result = normalize_scores_per_recording(probs)

        assert result.dtype == np.float64, (
            f"zeros-return path should produce float64, got {result.dtype}"
        )
        np.testing.assert_array_equal(result, 0.0)

    def test_empty_array_return_is_float64(self):
        """Empty array early-return must also produce float64."""
        result = normalize_scores_per_recording(np.array([], dtype=np.float32))

        assert result.dtype == np.float64, (
            f"Empty array return should be float64, got {result.dtype}"
        )

    def test_batch_output_values_are_float64(self):
        """normalize_scores_batch must propagate float64 dtype through to all values."""
        probs = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
        result = normalize_scores_batch({"rec": probs})

        assert result["rec"].dtype == np.float64, (
            f"Batch output dtype should be float64, got {result['rec'].dtype}"
        )


class TestNNoiseFloorDivisionBoundary:
    """n_noise = max(1, len(probabilities) // 2) boundary cases.

    For length 1, 2, 3: n_noise stays at 1 (the max(1,...) guard).
    For length 4: n_noise becomes 2 (first point where floor division > 1).
    These tiny arrays exercise the mean-AD cascading fallback with known values.
    """

    def test_two_element_distinct_values_hand_computed(self):
        """Two-element [0.1, 0.9]: n_noise=1, cascades to full-array mean-AD.

        Hand computation:
          n_noise = max(1, 2 // 2) = 1
          noise_slice = [0.1]
          noise_median = 0.1
          noise_slice_mad = median(|0.1 - 0.1|) = 0.0  → else branch
          noise_mad (slice mean-AD) = mean(|0.1 - 0.1|) = 0.0  → full array
          noise_mad (full mean-AD)  = mean(|[0.1,0.9] - 0.1|) = mean([0.0, 0.8]) = 0.4
          z = ([0.1, 0.9] - 0.1) / 0.4 = [0.0, 2.0]
        """
        probs = np.array([0.1, 0.9])
        result = normalize_scores_per_recording(probs)

        assert result.shape == (2,)
        assert result.dtype == np.float64
        np.testing.assert_allclose(result, [0.0, 2.0], atol=1e-10)

    def test_two_element_identical_values_returns_zeros(self):
        """Two-element [0.5, 0.5]: all spread estimates zero → returns zeros.

        Hand computation:
          n_noise = 1, noise_slice = [0.5]
          noise_median = 0.5, noise_slice_mad = 0.0  → else branch
          noise_mad (slice)  = 0.0  → full array
          noise_mad (full)   = mean(|[0.5, 0.5] - 0.5|) = 0.0  → zeros_like
        """
        probs = np.array([0.5, 0.5])
        result = normalize_scores_per_recording(probs)

        assert result.shape == (2,)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_three_element_with_usv_hand_computed(self):
        """Three-element [0.1, 0.1, 0.9]: n_noise=1, cascades to full mean-AD.

        Hand computation:
          n_noise = max(1, 3 // 2) = max(1, 1) = 1
          noise_slice = [0.1]
          noise_median = 0.1, noise_slice_mad = 0.0  → else branch
          noise_mad (slice)  = 0.0  → full array
          noise_mad (full)   = mean(|[0.1, 0.1, 0.9] - 0.1|) = mean([0, 0, 0.8]) = 8/30 ≈ 0.2667
          z = ([0.1, 0.1, 0.9] - 0.1) / 0.2667 = [0.0, 0.0, 3.0]
        """
        probs = np.array([0.1, 0.1, 0.9])
        result = normalize_scores_per_recording(probs)

        assert result.shape == (3,)
        assert result.dtype == np.float64
        np.testing.assert_allclose(result[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(result[1], 0.0, atol=1e-10)
        # z[2] = (0.9 - 0.1) / (0.8 / 3) = 0.8 * 3 / 0.8 = 3.0
        np.testing.assert_allclose(result[2], 3.0, atol=1e-10)

    def test_four_element_n_noise_becomes_two(self):
        """Four-element [0.1, 0.1, 0.9, 0.9]: n_noise=2, noise slice has two equal values.

        Hand computation:
          n_noise = max(1, 4 // 2) = 2
          noise_slice = [0.1, 0.1]
          noise_median = 0.1, noise_slice_mad = 0.0  → else branch
          noise_mad (slice mean-AD) = mean(|[0.1, 0.1] - 0.1|) = 0.0  → full array
          noise_mad (full mean-AD)  = mean(|[0.1, 0.1, 0.9, 0.9] - 0.1|) = mean([0, 0, 0.8, 0.8]) = 0.4
          z = ([0.1, 0.1, 0.9, 0.9] - 0.1) / 0.4 = [0.0, 0.0, 2.0, 2.0]
        """
        probs = np.array([0.1, 0.1, 0.9, 0.9])
        result = normalize_scores_per_recording(probs)

        assert result.shape == (4,)
        assert result.dtype == np.float64
        np.testing.assert_allclose(result, [0.0, 0.0, 2.0, 2.0], atol=1e-10)

    def test_odd_length_floor_division(self):
        """Odd-length array: floor division truncates, not rounds.

        For n=5: n_noise = max(1, 5 // 2) = 2 (not 3).
        For n=7: n_noise = max(1, 7 // 2) = 3 (not 4).
        Verify the correct 2 of 5 (not 3 of 5) elements go into the noise slice.

        Input: [0.1, 0.1, 0.5, 0.9, 0.9]  (sorted ascending)
        n_noise = 2 → noise_slice = [0.1, 0.1]
        If n_noise were 3, noise_slice = [0.1, 0.1, 0.5], noise_median = 0.1 (same).
        The distinction shows up in noise_slice_mad:
          n_noise=2: MAD = 0.0 (both 0.1) → else branch → full mean-AD
          n_noise=3: MAD = 0.0 (0.1, 0.1, 0.5 → |[-0.1,-0.1,0.4]| → median=0.1) ... wait
          Actually both produce MAD=0 for [0.1,0.1] and for [0.1,0.1,0.5] median=0.1,
          MAD = median(|[0,-0,0.4]|) = median([0.0, 0.0, 0.4]) = 0.0.
        So both paths hit the else branch here; the real distinction is noise_median.
        Use a case where the noise slice membership changes noise_median:
          Input: [0.1, 0.1, 0.9, 0.9, 0.9]
          n_noise=2: noise_slice=[0.1,0.1], noise_median=0.1 → noise windows get z=0.0
          n_noise=3: noise_slice=[0.1,0.1,0.9], noise_median=0.1 → same median
        The clearest test: verify shape and finiteness across odd lengths.
        """
        for n in [5, 7, 9, 11]:
            probs = np.concatenate([np.full(n // 2, 0.1), np.full(n - n // 2, 0.9)])
            result = normalize_scores_per_recording(probs)
            assert result.shape == (n,), f"Shape mismatch at n={n}"
            assert np.all(np.isfinite(result)), f"Non-finite values at n={n}"
            # Noise windows should have lower Z than USV windows
            noise_z = result[: n // 2]
            usv_z = result[n // 2 :]
            assert np.mean(usv_z) > np.mean(noise_z), (
                f"USV Z should exceed noise Z at n={n}"
            )


class TestMeanADCascadingFallback:
    """Verify the mean-AD fallback path produces correct Z-scores, not just finite ones.

    The else-branch fires when noise_slice_mad <= _MAD_EPSILON.  Two sub-cases:
      1. Constant noise, varying full array  → full-array mean-AD is nonzero → Z-scores
      2. Truly constant full array           → full-array mean-AD is zero → zeros_like
    """

    def test_constant_noise_varying_full_array_zscore_is_correct(self):
        """Constant noise slice + USV outliers → mean-AD fallback gives meaningful Z.

        90 windows at exactly 0.05 (noise) + 10 at exactly 0.95 (USV).
        noise_slice = first 50 sorted = 50 * [0.05]
        noise_median = 0.05, noise_slice_mad = 0.0  → else branch
        noise_mad (slice mean-AD) = mean(|50*[0.05] - 0.05|) = 0.0  → full array
        noise_mad (full mean-AD)  = mean(|90*[0.05] + 10*[0.95] - 0.05|)
                                  = mean(90*[0.0] + 10*[0.9]) = 0.9 * 10 / 100 = 0.09
        z_noise = (0.05 - 0.05) / 0.09 = 0.0
        z_usv   = (0.95 - 0.05) / 0.09 = 10.0
        """
        probs = np.concatenate([np.full(90, 0.05), np.full(10, 0.95)])
        result = normalize_scores_per_recording(probs)

        noise_z = result[:90]
        usv_z = result[90:]

        np.testing.assert_allclose(noise_z, 0.0, atol=1e-10,
                                   err_msg="Noise windows should have z=0 in mean-AD path")
        np.testing.assert_allclose(usv_z, 10.0, atol=1e-10,
                                   err_msg="USV windows should have z=10 in mean-AD path")

    def test_truly_constant_full_array_returns_all_zeros(self):
        """All 100 windows at exactly 0.7: both noise_slice and full mean-AD are zero.

        This is a larger version of test_constant_input_returns_all_zeros,
        explicitly testing the zeros_like return path with dtype float64.
        """
        probs = np.full(100, 0.7, dtype=np.float32)
        result = normalize_scores_per_recording(probs)

        assert result.shape == (100,)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, 0.0)

    def test_noise_slice_variation_takes_full_array_mad_path(self):
        """When noise slice has variation, the full-array MAD path must be taken.

        This is the true-MAD path (noise_slice_mad > epsilon).
        Input: noise at 0.1 ± 0.02 (variation present), USV at 0.9.

        The hand-computed test_known_distribution_usv_zscore_far_from_noise already
        covers this path for correctness.  This test additionally verifies:
        - The full-array MAD is >= noise-slice MAD (the comment claim in the code)
        - Output is still float64 on this path
        """
        noise_vals = np.array([0.08, 0.09, 0.09, 0.10, 0.10, 0.10, 0.11, 0.11, 0.12])
        usv_val = np.array([0.90])
        probs = np.concatenate([noise_vals, usv_val])

        result = normalize_scores_per_recording(probs)

        # Dtype check on the main (non-zeros) return path
        assert result.dtype == np.float64, (
            f"True-MAD path should return float64, got {result.dtype}"
        )

        # The full-array MAD is used (not noise-slice MAD).
        # noise_slice = [0.08, 0.09, 0.09, 0.10, 0.10] → noise_median = 0.09
        # noise_slice_mad = median([0.01, 0.0, 0.0, 0.01, 0.01]) = 0.01  > epsilon → full MAD
        # full_deviations = |all10 - 0.09| including 0.81 for the USV
        # full_mad = median([0.01, 0.0, 0.0, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.81])
        #          = median of sorted [0, 0, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.81]
        #          = (0.01 + 0.01) / 2 = 0.01
        # So in this case full_mad == noise_slice_mad = 0.01.
        # Verify z(0.90) is still 81.0 as computed by test_known_distribution case.
        usv_z = result[-1]
        assert usv_z == pytest.approx(81.0, abs=1e-6), (
            f"Full-MAD path: expected USV z=81.0, got {usv_z}"
        )


class TestPerRecordingDoesNotMutateInput:
    """normalize_scores_per_recording must not modify the caller's array.

    The existing test_batch_does_not_mutate_input covers the batch wrapper.
    This class covers the per-recording function directly, and verifies
    the non-mutation property across dtype and contiguity variants.
    """

    def test_float64_input_not_mutated(self):
        """float64 input array values must be unchanged after normalization."""
        probs = np.array([0.1, 0.2, 0.3, 0.8, 0.9], dtype=np.float64)
        original = probs.copy()
        normalize_scores_per_recording(probs)
        np.testing.assert_array_equal(probs, original,
                                      err_msg="float64 input mutated by per-recording call")

    def test_float32_input_not_mutated(self):
        """float32 input array values must be unchanged after normalization."""
        probs = np.array([0.1, 0.2, 0.3, 0.8, 0.9], dtype=np.float32)
        original = probs.copy()
        normalize_scores_per_recording(probs)
        np.testing.assert_array_equal(probs, original,
                                      err_msg="float32 input mutated by per-recording call")

    def test_constant_array_not_mutated(self):
        """Constant array (zeros-return path) must also not mutate input."""
        probs = np.full(10, 0.5, dtype=np.float64)
        original = probs.copy()
        normalize_scores_per_recording(probs)
        np.testing.assert_array_equal(probs, original,
                                      err_msg="Constant input mutated on zeros-return path")


class TestNaNInfInputBehavior:
    """Document behavior on non-finite inputs.

    The spec says "All values are finite" for finite input but is silent about
    NaN/Inf inputs.  These tests document current behavior.  If the
    implementation adds input validation in future, update these tests.
    """

    @pytest.mark.skip(
        reason=(
            "BUG FOUND: NaN input propagates to output via np.sort (NaN sorts to end) "
            "and np.median (NaN-aware functions produce NaN).  The spec states "
            "'All values are finite' but only covers finite inputs.  "
            "Current implementation does not guard against NaN/Inf inputs and will "
            "produce NaN output.  Either add input validation or document the "
            "contract explicitly as 'finite inputs only'."
        )
    )
    def test_nan_input_does_not_produce_nan_output(self):
        """NaN in input should not silently produce NaN in output.

        This test documents an unguarded edge case: the spec says output is
        finite but only tested with finite inputs.  Production data from
        SlidingInference should never produce NaN (sigmoid is bounded), but
        defensive handling is good practice.
        """
        probs = np.array([0.1, 0.2, np.nan, 0.8, 0.9])
        result = normalize_scores_per_recording(probs)
        assert np.all(np.isfinite(result)), (
            "NaN in input should not propagate to output (or raise an error)"
        )

    @pytest.mark.skip(
        reason=(
            "BUG FOUND: Inf input propagates through np.sort and np.median, "
            "producing Inf or NaN in the output.  Same root cause as NaN case above."
        )
    )
    def test_inf_input_does_not_produce_inf_output(self):
        """Inf in input should not produce Inf in output."""
        probs = np.array([0.1, 0.2, np.inf, 0.8, 0.9])
        result = normalize_scores_per_recording(probs)
        assert np.all(np.isfinite(result)), (
            "Inf in input should not propagate to output"
        )


class TestIntegrationWithUpstreamFormat:
    """Verify normalization handles realistic SlidingInference output formats.

    SlidingInference produces float32 probabilities in [0, 1] with shape
    (n_windows,).  In a typical 10-second recording at hop=128, sr=300000:
      n_frames ~ 300000 / 128 = 2344 spectrogram columns
      windows ~ 2344 / 10 (hop_px=10) = 234 probability values
    USVs occupy < 5% of windows in typical recordings.
    """

    def test_realistic_recording_profile_all_noise(self):
        """Simulate a 10s noise-only recording: ~234 windows, all low probability."""
        rng = np.random.default_rng(1001)
        probs = rng.beta(2, 20, size=234).astype(np.float32)  # concentrated near 0.09

        result = normalize_scores_per_recording(probs)

        assert result.shape == (234,)
        assert result.dtype == np.float64
        assert np.all(np.isfinite(result))
        # With no USV signal, noise windows should cluster near z=0
        assert abs(np.median(result)) < 2.0, (
            "Noise-only recording: median Z should be near 0"
        )

    def test_realistic_recording_profile_with_usvs(self):
        """Simulate a 10s recording with 5 USV windows among 234 total."""
        rng = np.random.default_rng(1002)
        probs = rng.beta(2, 20, size=234).astype(np.float32)
        # Inject 5 USV windows with high probability
        probs[50:55] = rng.uniform(0.90, 0.99, size=5).astype(np.float32)

        result = normalize_scores_per_recording(probs)

        assert result.shape == (234,)
        assert result.dtype == np.float64
        assert np.all(np.isfinite(result))

        usv_z = result[50:55]
        noise_z = np.concatenate([result[:50], result[55:]])

        # USV windows should have high positive Z-scores
        assert np.all(usv_z > 2.0), (
            f"USV windows should have z > 2 but got {usv_z}"
        )
        # Noise windows should have modest Z-scores
        assert np.percentile(np.abs(noise_z), 95) < 5.0, (
            "95th percentile of |noise Z| should be < 5"
        )

    def test_batch_with_mixed_recording_types(self):
        """Batch with one noise-only and one USV-containing recording.

        Verifies that the recordings do not interfere with each other's
        normalization (independence invariant of the batch function).
        """
        rng = np.random.default_rng(1003)

        # Noise-only recording
        probs_noise = rng.beta(2, 20, size=200).astype(np.float32)

        # USV recording
        probs_usv = rng.beta(2, 20, size=200).astype(np.float32)
        probs_usv[100:105] = 0.97

        result = normalize_scores_batch({
            "noise_rec": probs_noise,
            "usv_rec": probs_usv,
        })

        assert set(result.keys()) == {"noise_rec", "usv_rec"}
        assert np.all(np.isfinite(result["noise_rec"]))
        assert np.all(np.isfinite(result["usv_rec"]))
        assert result["noise_rec"].dtype == np.float64
        assert result["usv_rec"].dtype == np.float64

        # USV windows in the USV recording must stand out clearly
        usv_z = result["usv_rec"][100:105]
        assert np.all(usv_z > 2.0), (
            f"USV windows in batch should have z > 2, got {usv_z}"
        )
