"""Tests for Parameter Lab metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.param_lab.metrics import compute_metrics


class TestMetrics(unittest.TestCase):
    def test_compute_metrics_returns_expected_keys(self) -> None:
        spec_db = np.random.randn(50, 100).astype(np.float32) * 10.0 - 60.0
        freqs_hz = np.linspace(30_000.0, 125_000.0, 50)
        times_s = np.linspace(0.0, 1.0, 100)

        metrics = compute_metrics(spec_db, freqs_hz, times_s)

        expected_keys = {
            "noise_floor_db",
            "median_db",
            "p95_db",
            "p5_db",
            "contrast_db",
            "mean_db",
            "std_db",
            "max_db",
            "min_db",
            "frames",
            "freq_bins",
            "duration_s",
            "freq_min_hz",
            "freq_max_hz",
        }

        self.assertEqual(set(metrics.keys()), expected_keys)

    def test_compute_metrics_with_known_values(self) -> None:
        # Create spectrogram with known statistics
        spec_db = np.full((10, 20), -50.0, dtype=np.float32)
        freqs_hz = np.linspace(30_000.0, 90_000.0, 10)
        times_s = np.linspace(0.0, 2.0, 20)

        metrics = compute_metrics(spec_db, freqs_hz, times_s)

        # All values should be -50.0 dB
        self.assertAlmostEqual(metrics["mean_db"], -50.0, places=5)
        self.assertAlmostEqual(metrics["median_db"], -50.0, places=5)
        self.assertAlmostEqual(metrics["max_db"], -50.0, places=5)
        self.assertAlmostEqual(metrics["min_db"], -50.0, places=5)
        self.assertAlmostEqual(metrics["std_db"], 0.0, places=5)
        self.assertAlmostEqual(metrics["contrast_db"], 0.0, places=5)

        # Counts and ranges
        self.assertEqual(metrics["frames"], 20.0)
        self.assertEqual(metrics["freq_bins"], 10.0)
        self.assertAlmostEqual(metrics["duration_s"], 2.0, places=5)
        self.assertAlmostEqual(metrics["freq_min_hz"], 30_000.0, places=1)
        self.assertAlmostEqual(metrics["freq_max_hz"], 90_000.0, places=1)

    def test_compute_metrics_contrast_calculation(self) -> None:
        # Create spectrogram with known percentile values
        spec_db = np.linspace(-80.0, -20.0, 100).reshape(10, 10).astype(np.float32)
        freqs_hz = np.linspace(30_000.0, 125_000.0, 10)
        times_s = np.linspace(0.0, 0.5, 10)

        metrics = compute_metrics(spec_db, freqs_hz, times_s)

        # Contrast should be p95 - p5
        expected_contrast = metrics["p95_db"] - metrics["p5_db"]
        self.assertAlmostEqual(metrics["contrast_db"], expected_contrast, places=5)

        # For linearly spaced values, contrast should be significant
        self.assertGreater(metrics["contrast_db"], 40.0)

    def test_compute_metrics_noise_floor_is_p10(self) -> None:
        # Create data where we know the 10th percentile
        spec_db = np.array([
            -90, -85, -80, -75, -70, -65, -60, -55, -50, -45, -40
        ], dtype=np.float32).reshape(1, 11)
        freqs_hz = np.array([60_000.0])
        times_s = np.linspace(0.0, 0.1, 11)

        metrics = compute_metrics(spec_db, freqs_hz, times_s)

        # 10th percentile of sorted values
        expected_p10 = np.percentile(spec_db, 10.0)
        self.assertAlmostEqual(metrics["noise_floor_db"], expected_p10, places=5)

    def test_compute_metrics_empty_spectrogram_handling(self) -> None:
        spec_db = np.empty((0, 0), dtype=np.float32)
        freqs_hz = np.empty(0, dtype=np.float32)
        times_s = np.empty(0, dtype=np.float32)

        metrics = compute_metrics(spec_db, freqs_hz, times_s)

        # All metrics should be 0.0 for empty input
        for key, value in metrics.items():
            self.assertEqual(value, 0.0, f"Expected {key} to be 0.0, got {value}")

    def test_compute_metrics_single_frame(self) -> None:
        spec_db = np.random.randn(50, 1).astype(np.float32) * 10.0 - 60.0
        freqs_hz = np.linspace(30_000.0, 125_000.0, 50)
        times_s = np.array([0.5])

        metrics = compute_metrics(spec_db, freqs_hz, times_s)

        self.assertEqual(metrics["frames"], 1.0)
        # Duration for single frame is 0
        self.assertEqual(metrics["duration_s"], 0.0)

    def test_compute_metrics_duration_calculation(self) -> None:
        spec_db = np.random.randn(10, 100).astype(np.float32) * 10.0 - 60.0
        freqs_hz = np.linspace(30_000.0, 125_000.0, 10)
        times_s = np.linspace(1.5, 3.5, 100)

        metrics = compute_metrics(spec_db, freqs_hz, times_s)

        # Duration should be last - first
        expected_duration = times_s[-1] - times_s[0]
        self.assertAlmostEqual(metrics["duration_s"], expected_duration, places=5)

    def test_compute_metrics_all_values_finite(self) -> None:
        spec_db = np.random.randn(80, 150).astype(np.float32) * 15.0 - 55.0
        freqs_hz = np.linspace(30_000.0, 125_000.0, 80)
        times_s = np.linspace(0.0, 1.5, 150)

        metrics = compute_metrics(spec_db, freqs_hz, times_s)

        # All metric values should be finite
        for key, value in metrics.items():
            self.assertTrue(np.isfinite(value), f"{key} is not finite: {value}")

    def test_compute_metrics_percentiles_ordered(self) -> None:
        spec_db = np.random.randn(60, 120).astype(np.float32) * 12.0 - 58.0
        freqs_hz = np.linspace(30_000.0, 125_000.0, 60)
        times_s = np.linspace(0.0, 1.2, 120)

        metrics = compute_metrics(spec_db, freqs_hz, times_s)

        # p5 should be less than median, median less than p95
        self.assertLess(metrics["p5_db"], metrics["median_db"])
        self.assertLess(metrics["median_db"], metrics["p95_db"])

        # min should be <= p5, p95 should be <= max
        self.assertLessEqual(metrics["min_db"], metrics["p5_db"])
        self.assertLessEqual(metrics["p95_db"], metrics["max_db"])


if __name__ == "__main__":
    unittest.main()
