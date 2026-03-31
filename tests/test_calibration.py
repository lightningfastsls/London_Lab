"""Tests for temperature scaling calibration module."""

import json

import numpy as np
import pytest

from usv_spectrogram.postprocessing.calibration import (
    TemperatureScaler,
    compute_ece,
)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class TestTemperatureScaler:
    def test_identity_at_temperature_one(self):
        """T=1 calibrate should equal raw sigmoid."""
        logits = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        scaler = TemperatureScaler(temperature=1.0)
        result = scaler.calibrate(logits)
        expected = _sigmoid(logits)
        np.testing.assert_allclose(result, expected, atol=1e-7)

    def test_softening_high_temperature(self):
        """T=3 should move probabilities toward 0.5."""
        logits = np.array([-3.0, 3.0])
        scaler = TemperatureScaler(temperature=3.0)
        calibrated = scaler.calibrate(logits)
        raw = _sigmoid(logits)

        # Calibrated probs should be closer to 0.5 than raw
        assert abs(calibrated[0] - 0.5) < abs(raw[0] - 0.5)
        assert abs(calibrated[1] - 0.5) < abs(raw[1] - 0.5)

    def test_sharpening_low_temperature(self):
        """T=0.5 should move probabilities away from 0.5."""
        logits = np.array([-2.0, 2.0])
        scaler = TemperatureScaler(temperature=0.5)
        calibrated = scaler.calibrate(logits)
        raw = _sigmoid(logits)

        # Calibrated probs should be further from 0.5 than raw
        assert abs(calibrated[0] - 0.5) > abs(raw[0] - 0.5)
        assert abs(calibrated[1] - 0.5) > abs(raw[1] - 0.5)

    def test_fit_reduces_nll(self):
        """Fitting on synthetic data should not increase NLL."""
        rng = np.random.default_rng(42)
        # Create miscalibrated logits: overconfident positive class
        labels = rng.integers(0, 2, size=200).astype(np.float64)
        logits = labels * 4.0 - 2.0 + rng.normal(0, 0.5, size=200)

        scaler = TemperatureScaler()
        scaler.fit(logits, labels)

        assert scaler.fitted is True
        assert scaler.nll_after <= scaler.nll_before + 1e-10

    def test_save_load_roundtrip(self, tmp_path):
        """JSON save/load should preserve all fields."""
        scaler = TemperatureScaler(temperature=1.5)
        scaler.fitted = True
        scaler.nll_before = 0.7
        scaler.nll_after = 0.5

        path = tmp_path / "temperature.json"
        scaler.save(path)
        loaded = TemperatureScaler.load(path)

        assert loaded.temperature == pytest.approx(1.5)
        assert loaded.fitted is True
        assert loaded.nll_before == pytest.approx(0.7)
        assert loaded.nll_after == pytest.approx(0.5)

    def test_temperature_must_be_positive(self):
        """Negative or zero temperature should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            TemperatureScaler(temperature=-1.0)
        with pytest.raises(ValueError, match="positive"):
            TemperatureScaler(temperature=0.0)


    def test_fit_rejects_shape_mismatch(self):
        """Mismatched logits/labels shapes should raise ValueError."""
        scaler = TemperatureScaler()
        with pytest.raises(ValueError, match="same shape"):
            scaler.fit(np.array([1.0, 2.0]), np.array([1.0]))


class TestInferenceResultBackwardCompat:
    def test_backward_compat_inference_result(self):
        """InferenceResult without logits kwarg should default to None."""
        from usv_spectrogram.app.core.sliding_inference import InferenceResult

        result = InferenceResult(
            probabilities=np.array([0.5]),
            column_indices=np.array([10]),
            times=np.array([0.1]),
        )
        assert result.logits is None

    def test_logits_shape_matches_probabilities(self):
        """InferenceResult.logits should have same shape as probabilities."""
        from usv_spectrogram.app.core.sliding_inference import InferenceResult

        n = 50
        result = InferenceResult(
            probabilities=np.zeros(n),
            column_indices=np.arange(n),
            times=np.linspace(0, 1, n),
            logits=np.zeros(n),
        )
        assert result.logits.shape == result.probabilities.shape


class TestComputeECE:
    def test_ece_perfect_calibration(self):
        """Perfectly calibrated predictions should have ECE near 0."""
        rng = np.random.default_rng(123)
        n = 10000
        # Generate probabilities, then sample labels from Bernoulli(p)
        probs = rng.uniform(0.0, 1.0, size=n)
        labels = (rng.random(n) < probs).astype(float)

        ece = compute_ece(probs, labels, n_bins=15)
        # With enough samples, ECE of truly calibrated probs should be small
        assert ece < 0.03

    def test_ece_poorly_calibrated(self):
        """Overconfident predictions should have high ECE."""
        # All predictions at 0.9 but only 50% are actually positive
        probs = np.full(1000, 0.9)
        labels = np.zeros(1000)
        labels[:500] = 1.0

        ece = compute_ece(probs, labels)
        # |0.5 - 0.9| = 0.4, ECE should be large
        assert ece > 0.3
