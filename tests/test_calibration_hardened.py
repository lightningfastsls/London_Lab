"""Adversarial / edge-case tests for temperature scaling calibration module.

These tests supplement tests/test_calibration.py with paths the original
tests do not cover.  They were written by the test-hardener agent after
reviewing the implementation, the existing test suite, and the master-reviewer
findings in docs/reviews/calibration-review.md.

Coverage targets:
  A. ROADMAP test-plan items that were absent from the original suite
  B. Untested code paths (every branch/early-return)
  C. Numerical edge cases (NaN, Inf, extreme logits, boundary temperatures)
  D. Behavioural correctness (monotonicity, output range, round-trip)
  E. Integration boundary (compute_ece edge cases, save/load path coverage)
  F. Input validation
  G. Unfitted-scaler warning (S-1 from review)
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from usv_spectrogram.postprocessing.calibration import (
    TemperatureScaler,
    _binary_nll,
    compute_ece,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _make_miscalibrated(rng: np.random.Generator, n: int = 300):
    """Return (logits, labels) where model is overconfident.

    Labels are random; logits are high-magnitude and partially predictive
    but not perfectly separating.  This creates genuine miscalibration where
    temperature scaling yields a meaningful NLL improvement.
    """
    labels = rng.integers(0, 2, size=n).astype(np.float64)
    logits = labels * 6.0 - 3.0 + rng.normal(0, 0.3, size=n)
    return logits, labels


def _make_strongly_miscalibrated(rng: np.random.Generator, n: int = 400):
    """Return (logits, labels) with severe overconfidence.

    Model outputs logit=+4 for ALL positive examples and logit=-4 for ALL
    negative examples (zero within-class noise).  When the base rate is 50/50
    and logit magnitude is large, T=1 gives near-zero NLL because the model
    is correct.  To get a *large* improvement we instead assign logits that
    are confidently RIGHT about direction but with random label noise mixed in
    — creating a model that is overconfident relative to true accuracy.
    """
    # True labels are random
    labels = rng.integers(0, 2, size=n).astype(np.float64)
    # Model is overconfident: assigns very high positive logit to ~70% of
    # samples regardless of true label, very negative to the rest.
    # This means confidence >> accuracy → large calibration error.
    model_prediction = rng.choice([1, 0], size=n, p=[0.7, 0.3]).astype(float)
    logits = model_prediction * 8.0 - 4.0  # +4 or -4
    return logits, labels


# ===========================================================================
# A. ROADMAP test-plan gaps
# ===========================================================================

class TestRoadmapGaps:
    """Items from ROADMAP §15.3 test plan not covered by the original suite."""

    def test_calibrate_unfitted_returns_raw_sigmoid(self):
        """calibrate() on an unfitted scaler with T=1.5 should still apply T.

        ROADMAP item 1 checks T=1.0 identity; there is no test for the default
        T=1.5 path on an *unfitted* scaler.  The scaler must still honour the
        constructor temperature even before fit() is called.
        """
        scaler = TemperatureScaler(temperature=1.5)
        logits = np.array([-3.0, 0.0, 3.0])
        result = scaler.calibrate(logits)
        expected = _sigmoid(logits / 1.5)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_fit_stores_nll_before_and_after(self):
        """fit() must populate both nll_before and nll_after (ROADMAP item 4)."""
        rng = np.random.default_rng(7)
        logits, labels = _make_miscalibrated(rng)
        scaler = TemperatureScaler()
        scaler.fit(logits, labels)

        assert scaler.nll_before is not None
        assert scaler.nll_after is not None
        assert isinstance(scaler.nll_before, float)
        assert isinstance(scaler.nll_after, float)

    def test_default_temperature_is_1_5(self):
        """ROADMAP specifies default temperature=1.5 (fixed after W-4 review)."""
        scaler = TemperatureScaler()
        assert scaler.temperature == pytest.approx(1.5)

    def test_fit_uses_self_temperature_as_initial_guess(self):
        """x0 in L-BFGS-B should start from self.temperature (W-4 fix).

        Verify that the optimisation succeeds and nll_after <= nll_before
        regardless of the starting temperature, which exercises the x0 path.
        """
        rng = np.random.default_rng(99)
        logits, labels = _make_miscalibrated(rng)

        for t0 in [0.5, 1.5, 5.0]:
            scaler = TemperatureScaler(temperature=t0)
            scaler.fit(logits, labels)
            assert scaler.nll_after <= scaler.nll_before + 1e-8, (
                f"Starting at T={t0}: nll_after {scaler.nll_after} > "
                f"nll_before {scaler.nll_before}"
            )


# ===========================================================================
# B. Untested code paths
# ===========================================================================

class TestCodePaths:
    """Every branch in calibration.py that the original suite skips."""

    # --- __post_init__ negative branch already covered; test zero separately ---

    def test_temperature_zero_raises(self):
        """T=0 is rejected by __post_init__ (separate from T<0 test)."""
        with pytest.raises(ValueError, match="positive"):
            TemperatureScaler(temperature=0.0)

    # --- _binary_nll directly ---

    def test_binary_nll_positive_label(self):
        """_binary_nll with label=1 for logit=0 should equal log(2)."""
        # -log(sigmoid(0)) = -log(0.5) = log(2)
        nll = _binary_nll(np.array([0.0]), np.array([1.0]), temperature=1.0)
        assert nll == pytest.approx(np.log(2), rel=1e-6)

    def test_binary_nll_negative_label(self):
        """_binary_nll with label=0 for logit=0 should equal log(2)."""
        # -log(1 - sigmoid(0)) = -log(0.5) = log(2)
        nll = _binary_nll(np.array([0.0]), np.array([0.0]), temperature=1.0)
        assert nll == pytest.approx(np.log(2), rel=1e-6)

    def test_binary_nll_temperature_scaling(self):
        """Doubling temperature is equivalent to halving the logit."""
        logits = np.array([1.0, -1.0, 2.0])
        labels = np.array([1.0, 0.0, 1.0])
        nll_t2 = _binary_nll(logits, labels, temperature=2.0)
        nll_halved = _binary_nll(logits / 2.0, labels, temperature=1.0)
        assert nll_t2 == pytest.approx(nll_halved, rel=1e-10)

    def test_binary_nll_nonnegative(self):
        """NLL should always be >= 0 for valid binary labels."""
        rng = np.random.default_rng(0)
        logits = rng.normal(0, 5, 100)
        labels = rng.integers(0, 2, 100).astype(float)
        nll = _binary_nll(logits, labels, temperature=1.0)
        assert nll >= 0.0

    # --- calibrate() output range ---

    def test_calibrate_output_in_unit_interval(self):
        """calibrate() must always return values in [0, 1]."""
        scaler = TemperatureScaler(temperature=1.5)
        # Cover extreme positive and negative logits
        logits = np.array([-1000.0, -100.0, -10.0, 0.0, 10.0, 100.0, 1000.0])
        result = scaler.calibrate(logits)
        assert np.all(result >= 0.0), "Probabilities below 0 found"
        assert np.all(result <= 1.0), "Probabilities above 1 found"

    # --- fit() with a single sample ---

    def test_fit_single_element(self):
        """fit() must not crash on a single (logit, label) pair."""
        scaler = TemperatureScaler()
        scaler.fit(np.array([2.0]), np.array([1.0]))
        assert scaler.fitted is True
        assert np.isfinite(scaler.temperature)

    # --- save() parent directory creation ---

    def test_save_creates_parent_directories(self, tmp_path):
        """save() must create intermediate directories if they don't exist."""
        scaler = TemperatureScaler(temperature=2.0)
        deep_path = tmp_path / "nested" / "dir" / "temperature.json"
        scaler.save(deep_path)
        assert deep_path.exists()

    # --- load() reads back exactly what save() wrote ---

    def test_load_json_structure(self, tmp_path):
        """Saved JSON must contain all four expected keys."""
        scaler = TemperatureScaler(temperature=1.23)
        scaler.nll_before = 0.693
        scaler.nll_after = 0.500
        scaler.fitted = True
        path = tmp_path / "t.json"
        scaler.save(path)
        data = json.loads(path.read_text())
        assert set(data.keys()) == {"temperature", "fitted", "nll_before", "nll_after"}

    def test_load_with_null_nll_fields(self, tmp_path):
        """load() must tolerate JSON null for nll_before / nll_after (unfitted)."""
        path = tmp_path / "t.json"
        path.write_text(json.dumps({
            "temperature": 1.5,
            "fitted": False,
            "nll_before": None,
            "nll_after": None,
        }))
        scaler = TemperatureScaler.load(path)
        assert scaler.nll_before is None
        assert scaler.nll_after is None
        assert scaler.fitted is False


# ===========================================================================
# C. Numerical edge cases
# ===========================================================================

class TestNumericalEdgeCases:
    """NaN, Inf, boundary temperatures, extreme logits."""

    def test_calibrate_zero_logit_is_half(self):
        """sigmoid(0 / T) == 0.5 for any positive T."""
        for T in [0.01, 0.5, 1.0, 1.5, 3.0, 50.0]:
            scaler = TemperatureScaler(temperature=T)
            result = scaler.calibrate(np.array([0.0]))
            assert result[0] == pytest.approx(0.5, abs=1e-10), (
                f"T={T}: sigmoid(0/T) should be 0.5, got {result[0]}"
            )

    def test_calibrate_very_large_positive_logit(self):
        """Very large positive logit should produce probability close to 1."""
        scaler = TemperatureScaler(temperature=1.0)
        result = scaler.calibrate(np.array([1000.0]))
        assert result[0] > 0.999

    def test_calibrate_very_large_negative_logit(self):
        """Very large negative logit should produce probability close to 0."""
        scaler = TemperatureScaler(temperature=1.0)
        # NOTE: this triggers a numpy overflow RuntimeWarning because the
        # implementation uses 1/(1+exp(-scaled)) directly.  The result is
        # still correct (0.0 due to float saturation) but the warning is a
        # latent code-quality issue.  See test_calibrate_overflow_warning below.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = scaler.calibrate(np.array([-1000.0]))
        assert result[0] < 0.001

    def test_calibrate_no_overflow_warning_for_extreme_negative_logits(self):
        """calibrate() must NOT emit RuntimeWarning for very negative logits."""
        scaler = TemperatureScaler(temperature=1.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scaler.calibrate(np.array([-1000.0]))
        overflow_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)
                             and "overflow" in str(x.message).lower()]
        assert len(overflow_warnings) == 0, (
            f"Expected no overflow warnings, got: {overflow_warnings}"
        )

    def test_calibrate_no_nan_for_extreme_logits(self):
        """calibrate() must not produce NaN on logits in [-1000, 1000]."""
        scaler = TemperatureScaler(temperature=1.5)
        logits = np.linspace(-1000.0, 1000.0, 201)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = scaler.calibrate(logits)
        assert np.all(np.isfinite(result)), "NaN or Inf in calibrate() output"

    def test_binary_nll_extreme_logits_stable(self):
        """_binary_nll should be finite even for very large logit magnitudes."""
        logits = np.array([-1000.0, 1000.0])
        labels = np.array([0.0, 1.0])
        nll = _binary_nll(logits, labels, temperature=1.0)
        assert np.isfinite(nll), f"_binary_nll returned non-finite: {nll}"

    def test_binary_nll_exact_zero_logit(self):
        """_binary_nll at logit=0 is log(2) regardless of label sign."""
        for label in [0.0, 1.0]:
            nll = _binary_nll(np.array([0.0]), np.array([label]), temperature=1.0)
            assert nll == pytest.approx(np.log(2), rel=1e-6)

    def test_temperature_boundary_min(self):
        """Temperature=0.01 (optimizer lower bound) must not crash calibrate()."""
        scaler = TemperatureScaler(temperature=0.01)
        logits = np.array([-2.0, 0.0, 2.0])
        result = scaler.calibrate(logits)
        assert np.all(np.isfinite(result))

    def test_temperature_boundary_max(self):
        """Temperature=50.0 (optimizer upper bound) should flatten probabilities."""
        scaler = TemperatureScaler(temperature=50.0)
        logits = np.array([-10.0, 10.0])
        result = scaler.calibrate(logits)
        # At T=50, both should be very close to 0.5
        assert abs(result[0] - 0.5) < 0.15
        assert abs(result[1] - 0.5) < 0.15

    def test_fit_all_same_label_one(self):
        """fit() should not crash when all labels are 1 (degenerate dataset)."""
        rng = np.random.default_rng(1)
        logits = rng.normal(2.0, 1.0, 50)
        labels = np.ones(50)
        scaler = TemperatureScaler()
        scaler.fit(logits, labels)
        assert scaler.fitted is True
        assert np.isfinite(scaler.temperature)

    def test_fit_all_same_label_zero(self):
        """fit() should not crash when all labels are 0."""
        rng = np.random.default_rng(2)
        logits = rng.normal(-2.0, 1.0, 50)
        labels = np.zeros(50)
        scaler = TemperatureScaler()
        scaler.fit(logits, labels)
        assert scaler.fitted is True
        assert np.isfinite(scaler.temperature)

    def test_fit_perfectly_calibrated_model(self):
        """Perfectly calibrated logits → optimizer should return T close to 1."""
        rng = np.random.default_rng(3)
        n = 2000
        # Draw calibrated probs and binary labels from them
        probs = rng.uniform(0.1, 0.9, n)
        labels = (rng.random(n) < probs).astype(float)
        # Convert probs back to logits: logit = log(p/(1-p))
        logits = np.log(probs / (1.0 - probs))
        scaler = TemperatureScaler()
        scaler.fit(logits, labels)
        # A perfectly calibrated model needs no correction; T should be near 1.0
        assert 0.5 < scaler.temperature < 3.0, (
            f"Expected T near 1.0 for calibrated model, got {scaler.temperature}"
        )

    def test_save_load_boundary_temperatures(self, tmp_path):
        """save/load round-trip for boundary temperature values."""
        for T in [0.01, 50.0]:
            scaler = TemperatureScaler(temperature=T)
            path = tmp_path / f"temp_{T}.json"
            scaler.save(path)
            loaded = TemperatureScaler.load(path)
            assert loaded.temperature == pytest.approx(T, rel=1e-9)


# ===========================================================================
# D. Behavioural correctness
# ===========================================================================

class TestBehaviouralCorrectness:
    """Monotonicity, output range, round-trip improvement."""

    def test_calibrate_monotone_in_logits(self):
        """calibrate() must be a monotonically increasing function of logits."""
        scaler = TemperatureScaler(temperature=1.5)
        logits = np.linspace(-10.0, 10.0, 200)
        result = scaler.calibrate(logits)
        diffs = np.diff(result)
        assert np.all(diffs >= 0), "calibrate() is not monotonically non-decreasing"

    def test_calibrate_monotone_after_fit(self):
        """Monotonicity holds for any T returned by fit()."""
        rng = np.random.default_rng(10)
        logits_fit, labels = _make_miscalibrated(rng)
        scaler = TemperatureScaler()
        scaler.fit(logits_fit, labels)

        logits_test = np.linspace(-10.0, 10.0, 200)
        result = scaler.calibrate(logits_test)
        diffs = np.diff(result)
        assert np.all(diffs >= 0)

    def test_fit_nll_improvement_meaningful(self):
        """For a strongly miscalibrated model, NLL must improve by > 0.05.

        Uses _make_strongly_miscalibrated which generates logits that are
        confidently wrong about actual class probabilities (high-magnitude
        logits assigned nearly-at-random relative to true labels), ensuring
        a large calibration gap and thus a large NLL improvement.
        """
        rng = np.random.default_rng(20)
        logits, labels = _make_strongly_miscalibrated(rng)
        scaler = TemperatureScaler()
        scaler.fit(logits, labels)
        improvement = scaler.nll_before - scaler.nll_after
        assert improvement > 0.05, (
            f"Expected meaningful NLL improvement (>0.05), got {improvement:.4f}.  "
            f"nll_before={scaler.nll_before:.4f}, nll_after={scaler.nll_after:.4f}, "
            f"fitted_T={scaler.temperature:.4f}"
        )

    def test_calibrate_symmetric_at_zero_logit(self):
        """calibrate(+z) + calibrate(-z) should equal 1.0 (sigmoid symmetry)."""
        scaler = TemperatureScaler(temperature=2.0)
        z_vals = np.array([0.5, 1.0, 2.0, 5.0])
        pos = scaler.calibrate(z_vals)
        neg = scaler.calibrate(-z_vals)
        np.testing.assert_allclose(pos + neg, 1.0, atol=1e-12)

    def test_fit_then_calibrate_returns_finite_probs(self):
        """After fit(), calibrate() must return all-finite values."""
        rng = np.random.default_rng(30)
        logits_fit, labels = _make_miscalibrated(rng)
        scaler = TemperatureScaler()
        scaler.fit(logits_fit, labels)

        logits_test = rng.normal(0, 3, 1000)
        result = scaler.calibrate(logits_test)
        assert np.all(np.isfinite(result))

    def test_save_load_preserves_fitted_temperature_for_calibrate(self, tmp_path):
        """Loaded scaler must produce identical calibrate() output as original."""
        rng = np.random.default_rng(40)
        logits, labels = _make_miscalibrated(rng)
        scaler = TemperatureScaler()
        scaler.fit(logits, labels)

        path = tmp_path / "fitted.json"
        scaler.save(path)
        loaded = TemperatureScaler.load(path)

        test_logits = np.linspace(-5.0, 5.0, 50)
        np.testing.assert_allclose(
            scaler.calibrate(test_logits),
            loaded.calibrate(test_logits),
            atol=1e-12,
        )

    def test_large_array_does_not_crash(self):
        """calibrate() on a 1M-element array should complete without error."""
        scaler = TemperatureScaler(temperature=1.5)
        logits = np.random.default_rng(0).normal(0, 3, 1_000_000)
        result = scaler.calibrate(logits)
        assert result.shape == (1_000_000,)
        assert np.all(np.isfinite(result))

    def test_large_fit_does_not_crash(self):
        """fit() on 50k samples should complete without error."""
        rng = np.random.default_rng(50)
        logits, labels = _make_miscalibrated(rng, n=50_000)
        scaler = TemperatureScaler()
        scaler.fit(logits, labels)
        assert scaler.fitted is True


# ===========================================================================
# E. compute_ece edge cases
# ===========================================================================

class TestComputeECEEdgeCases:
    """Paths in compute_ece() not covered by the original suite."""

    def test_ece_prob_exactly_zero_included_in_first_bin(self):
        """prob == 0.0 must be caught by the first-bin special case."""
        # All predictions at exactly 0.0, labels all 0 → ECE=0 (perfect calibration)
        probs = np.zeros(100)
        labels = np.zeros(100)
        ece = compute_ece(probs, labels, n_bins=15)
        assert ece == pytest.approx(0.0, abs=1e-10)

    def test_ece_prob_exactly_zero_with_wrong_labels(self):
        """prob==0.0 but label==1.0 should give high ECE."""
        probs = np.zeros(100)
        labels = np.ones(100)
        ece = compute_ece(probs, labels, n_bins=15)
        # |accuracy - confidence| = |1.0 - 0.0| = 1.0
        assert ece > 0.9

    def test_ece_all_probs_at_one(self):
        """All predictions at 1.0 with all labels=1 → ECE=0."""
        probs = np.ones(100)
        labels = np.ones(100)
        ece = compute_ece(probs, labels, n_bins=15)
        assert ece == pytest.approx(0.0, abs=1e-10)

    def test_ece_all_probs_at_one_wrong_labels(self):
        """All predictions at 1.0 with all labels=0 → ECE close to 1."""
        probs = np.ones(100)
        labels = np.zeros(100)
        ece = compute_ece(probs, labels, n_bins=15)
        assert ece > 0.9

    def test_ece_single_sample(self):
        """Single sample should not crash."""
        ece = compute_ece(np.array([0.7]), np.array([1.0]), n_bins=15)
        assert np.isfinite(ece)

    def test_ece_one_bin(self):
        """n_bins=1 should treat the entire [0,1] range as a single bin."""
        rng = np.random.default_rng(5)
        n = 200
        probs = rng.uniform(0, 1, n)
        labels = (rng.random(n) < probs).astype(float)
        ece_1 = compute_ece(probs, labels, n_bins=1)
        assert np.isfinite(ece_1)
        assert 0.0 <= ece_1 <= 1.0

    def test_ece_empty_bins_skipped(self):
        """Bins with no predictions should not contribute to ECE (no NaN)."""
        # All predictions at exactly 0.5 → most bins empty
        probs = np.full(50, 0.5)
        labels = np.ones(50)
        ece = compute_ece(probs, labels, n_bins=15)
        assert np.isfinite(ece)

    def test_ece_returns_float(self):
        """compute_ece should always return a Python float, not numpy scalar."""
        probs = np.array([0.3, 0.7])
        labels = np.array([0.0, 1.0])
        result = compute_ece(probs, labels)
        assert isinstance(result, float)

    def test_ece_zero_bins_raises(self):
        """n_bins=0 should raise ValueError (S-2 from review)."""
        with pytest.raises(ValueError):
            compute_ece(np.array([0.5]), np.array([1.0]), n_bins=0)


# ===========================================================================
# F. Input validation – shape mismatches and wrong dtypes
# ===========================================================================

class TestInputValidation:
    """Shape mismatches and wrong types that should raise cleanly."""

    def test_fit_2d_logits_raises(self):
        """fit() with 2D logits and 1D labels should raise (shape mismatch)."""
        scaler = TemperatureScaler()
        with pytest.raises(ValueError, match="same shape"):
            scaler.fit(
                np.array([[1.0, 2.0]]),  # shape (1, 2)
                np.array([1.0, 0.0]),    # shape (2,)
            )

    def test_fit_labels_longer_than_logits_raises(self):
        """More labels than logits is a shape mismatch and must raise."""
        scaler = TemperatureScaler()
        with pytest.raises(ValueError, match="same shape"):
            scaler.fit(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0]))

    def test_calibrate_empty_array(self):
        """calibrate() on an empty array should return an empty array."""
        scaler = TemperatureScaler(temperature=1.5)
        result = scaler.calibrate(np.array([]))
        assert result.shape == (0,)

    def test_binary_nll_empty_arrays(self):
        """_binary_nll on empty arrays must not raise a hard exception.

        np.mean([]) returns nan with a RuntimeWarning — that is acceptable.
        What is not acceptable is a crash (ValueError, ZeroDivisionError, etc.).
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                val = _binary_nll(np.array([]), np.array([]), temperature=1.0)
                # nan is acceptable; just must not crash
                assert val is not None
            except (ValueError, ZeroDivisionError) as exc:
                pytest.fail(f"_binary_nll raised unexpectedly: {exc}")


# ===========================================================================
# G. Unfitted-scaler warning (S-1 from review)
# ===========================================================================

class TestUnfittedScaler:
    """S-1 from calibration-review.md: calibrate() on unfitted scaler."""

    def test_calibrate_unfitted_emits_warning(self):
        """calibrate() before fit() should emit a UserWarning."""
        scaler = TemperatureScaler()
        logits = np.array([0.0, 1.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scaler.calibrate(logits)
        assert len(w) >= 1
        assert any(issubclass(warning.category, UserWarning) for warning in w)

    def test_fitted_flag_is_false_on_construction(self):
        """Newly constructed scaler must have fitted=False."""
        scaler = TemperatureScaler()
        assert scaler.fitted is False

    def test_fitted_flag_is_true_after_fit(self):
        """fit() must set fitted=True."""
        rng = np.random.default_rng(60)
        logits, labels = _make_miscalibrated(rng)
        scaler = TemperatureScaler()
        scaler.fit(logits, labels)
        assert scaler.fitted is True
