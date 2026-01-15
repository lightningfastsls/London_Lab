"""Tests for STFT core utilities."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import signal

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram._stft_core import compute_stft_frames_db, extract_frames


class TestStftCore(unittest.TestCase):
    def test_extract_frames_output_shape(self) -> None:
        samples = np.arange(1000, dtype=np.float32)
        window_length = 256
        hop_length = 128

        frames = extract_frames(samples, window_length, hop_length)

        expected_n_frames = 1 + (1000 - 256) // 128
        self.assertEqual(frames.shape, (expected_n_frames, window_length))
        self.assertEqual(frames.shape[0], 6)

    def test_extract_frames_with_empty_input(self) -> None:
        samples = np.empty(0, dtype=np.float32)
        window_length = 256
        hop_length = 128

        frames = extract_frames(samples, window_length, hop_length)

        self.assertEqual(frames.shape, (0, window_length))

    def test_extract_frames_with_short_input(self) -> None:
        samples = np.arange(100, dtype=np.float32)
        window_length = 256
        hop_length = 128

        frames = extract_frames(samples, window_length, hop_length)

        # Not enough samples for even one frame
        self.assertEqual(frames.shape, (0, window_length))

    def test_extract_frames_exact_fit(self) -> None:
        samples = np.arange(256, dtype=np.float32)
        window_length = 256
        hop_length = 128

        frames = extract_frames(samples, window_length, hop_length)

        # Exactly one frame fits
        self.assertEqual(frames.shape, (1, window_length))
        np.testing.assert_array_equal(frames[0], samples)

    def test_extract_frames_content_correctness(self) -> None:
        samples = np.arange(10, dtype=np.float32)
        window_length = 4
        hop_length = 2

        frames = extract_frames(samples, window_length, hop_length)

        expected_n_frames = 1 + (10 - 4) // 2
        self.assertEqual(frames.shape[0], expected_n_frames)

        # Check each frame
        np.testing.assert_array_equal(frames[0], [0, 1, 2, 3])
        np.testing.assert_array_equal(frames[1], [2, 3, 4, 5])
        np.testing.assert_array_equal(frames[2], [4, 5, 6, 7])

    def test_compute_stft_frames_db_output_shape(self) -> None:
        n_frames = 10
        window_length = 256
        n_fft = 512

        frames = np.random.randn(n_frames, window_length).astype(np.float32)
        window = signal.get_window("hann", window_length, fftbins=True)
        freqs_hz = np.fft.rfftfreq(n_fft, d=1.0 / 250_000)
        band_mask = (freqs_hz >= 30_000.0) & (freqs_hz <= 125_000.0)
        eps = 1e-12

        spec_db = compute_stft_frames_db(frames, window, n_fft, band_mask, eps)

        expected_freq_bins = band_mask.sum()
        self.assertEqual(spec_db.shape, (expected_freq_bins, n_frames))

    def test_compute_stft_frames_db_with_known_input(self) -> None:
        # Create a simple sinusoid at a known frequency
        sample_rate_hz = 1000
        freq_hz = 100.0
        window_length = 256
        n_fft = 512
        n_frames = 1

        # Single frame of a pure tone
        t = np.arange(window_length) / sample_rate_hz
        tone = np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32)
        frames = tone.reshape(1, window_length)

        window = signal.get_window("hann", window_length, fftbins=True)
        freqs_hz = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate_hz)
        band_mask = (freqs_hz >= 0.0) & (freqs_hz <= 500.0)
        eps = 1e-12

        spec_db = compute_stft_frames_db(frames, window, n_fft, band_mask, eps)

        # The spectrum should have a peak at 100 Hz
        freq_bin_100hz = np.argmin(np.abs(freqs_hz[band_mask] - freq_hz))
        peak_db = spec_db[freq_bin_100hz, 0]

        # Find the mean of bins far from the peak
        bins_away_from_peak = np.abs(freqs_hz[band_mask] - freq_hz) > 50.0
        mean_far_db = np.mean(spec_db[bins_away_from_peak, 0])

        # Peak should be significantly above the background
        self.assertGreater(peak_db - mean_far_db, 10.0)

    def test_compute_stft_frames_db_avoids_log_zero(self) -> None:
        # Test with all-zero input to ensure eps prevents log(0)
        n_frames = 5
        window_length = 128
        n_fft = 256

        frames = np.zeros((n_frames, window_length), dtype=np.float32)
        window = signal.get_window("hann", window_length, fftbins=True)
        freqs_hz = np.fft.rfftfreq(n_fft, d=1.0 / 250_000)
        band_mask = (freqs_hz >= 30_000.0) & (freqs_hz <= 125_000.0)
        eps = 1e-12

        spec_db = compute_stft_frames_db(frames, window, n_fft, band_mask, eps)

        # Should produce finite dB values (not -inf or nan)
        self.assertTrue(np.all(np.isfinite(spec_db)))

        # All values should be the same (log of eps)
        expected_db = 20.0 * np.log10(eps)
        np.testing.assert_allclose(spec_db, expected_db, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
