"""Tests for Zarr storage utilities."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.config import SpectrogramConfig
from usv_spectrogram.storage_zarr import (
    FREQS_KEY,
    SPECTROGRAM_KEY,
    TIMES_KEY,
    append_spectrogram_chunk,
    init_spectrogram_store,
)


class TestStorageZarr(unittest.TestCase):
    def test_init_spectrogram_store_creates_store_with_correct_arrays(self) -> None:
        cfg = SpectrogramConfig()
        freqs_hz = np.linspace(30_000.0, 125_000.0, 100)
        sample_rate_hz = 250_000

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.zarr"

            group = init_spectrogram_store(store_path, freqs_hz, sample_rate_hz, cfg)

            # Check that group was created
            self.assertIsInstance(group, zarr.Group)

            # Check attributes
            self.assertEqual(group.attrs["sample_rate_hz"], sample_rate_hz)
            self.assertEqual(group.attrs["window_length"], cfg.window_length)
            self.assertEqual(group.attrs["n_fft"], cfg.n_fft())
            self.assertEqual(group.attrs["f_min_hz"], cfg.f_min_hz)
            self.assertEqual(group.attrs["f_max_hz"], cfg.f_max_hz)

            # Check arrays exist
            self.assertIn(FREQS_KEY, group)
            self.assertIn(TIMES_KEY, group)
            self.assertIn(SPECTROGRAM_KEY, group)

            # Check freqs_hz array
            np.testing.assert_allclose(group[FREQS_KEY][:], freqs_hz, rtol=1e-6)

            # Check times_s array (empty initially)
            self.assertEqual(group[TIMES_KEY].shape, (0,))

            # Check spectrogram array (empty initially)
            spec_shape = group[SPECTROGRAM_KEY].shape
            self.assertEqual(spec_shape, (100, 0))

    def test_append_spectrogram_chunk_writes_data(self) -> None:
        cfg = SpectrogramConfig()
        freqs_hz = np.linspace(30_000.0, 125_000.0, 100)
        sample_rate_hz = 250_000

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "append_test.zarr"
            group = init_spectrogram_store(store_path, freqs_hz, sample_rate_hz, cfg)

            # Create chunk to append
            n_freq_bins = 100
            n_frames = 50
            spec_db = np.random.randn(n_freq_bins, n_frames).astype(np.float32) * 10.0 - 60.0
            times_s = np.linspace(0.0, 0.1, n_frames)

            append_spectrogram_chunk(group, spec_db, times_s)

            # Check that data was written
            self.assertEqual(group[SPECTROGRAM_KEY].shape, (n_freq_bins, n_frames))
            self.assertEqual(group[TIMES_KEY].shape, (n_frames,))
            np.testing.assert_allclose(group[TIMES_KEY][:], times_s, rtol=1e-6)

    def test_append_spectrogram_chunk_multiple_writes(self) -> None:
        cfg = SpectrogramConfig()
        freqs_hz = np.linspace(30_000.0, 125_000.0, 80)
        sample_rate_hz = 250_000

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "multi_append.zarr"
            group = init_spectrogram_store(store_path, freqs_hz, sample_rate_hz, cfg)

            n_freq_bins = 80

            # First chunk
            chunk1 = np.random.randn(n_freq_bins, 30).astype(np.float32) * 10.0 - 60.0
            times1 = np.linspace(0.0, 0.05, 30)
            append_spectrogram_chunk(group, chunk1, times1)

            # Second chunk
            chunk2 = np.random.randn(n_freq_bins, 40).astype(np.float32) * 10.0 - 60.0
            times2 = np.linspace(0.05, 0.10, 40)
            append_spectrogram_chunk(group, chunk2, times2)

            # Check final shape
            expected_frames = 30 + 40
            self.assertEqual(group[SPECTROGRAM_KEY].shape, (n_freq_bins, expected_frames))
            self.assertEqual(group[TIMES_KEY].shape, (expected_frames,))

    def test_roundtrip_write_then_read_back(self) -> None:
        cfg = SpectrogramConfig()
        freqs_hz = np.linspace(30_000.0, 125_000.0, 120)
        sample_rate_hz = 250_000

        # Original data
        n_freq_bins = 120
        n_frames = 100
        spec_db_original = np.random.randn(n_freq_bins, n_frames).astype(np.float32) * 10.0 - 50.0
        times_s_original = np.linspace(0.0, 0.2, n_frames)

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "roundtrip.zarr"

            # Write
            group = init_spectrogram_store(store_path, freqs_hz, sample_rate_hz, cfg)
            append_spectrogram_chunk(group, spec_db_original, times_s_original)

            # Read back
            group_read = zarr.open_group(str(store_path), mode="r")
            spec_int16 = group_read[SPECTROGRAM_KEY][:]
            times_read = group_read[TIMES_KEY][:]
            freqs_read = group_read[FREQS_KEY][:]

            # Check times and freqs are exact
            np.testing.assert_allclose(times_read, times_s_original, rtol=1e-6)
            np.testing.assert_allclose(freqs_read, freqs_hz, rtol=1e-6)

            # Check spectrogram is within quantization tolerance
            # Storage uses 0.1 dB per LSB
            scale_db_per_lsb = 0.1
            spec_db_read = spec_int16.astype(np.float32) * scale_db_per_lsb
            np.testing.assert_allclose(spec_db_read, spec_db_original, atol=scale_db_per_lsb)

    def test_append_empty_chunk_does_nothing(self) -> None:
        cfg = SpectrogramConfig()
        freqs_hz = np.linspace(30_000.0, 125_000.0, 60)
        sample_rate_hz = 250_000

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "empty_chunk.zarr"
            group = init_spectrogram_store(store_path, freqs_hz, sample_rate_hz, cfg)

            # Append empty chunk
            empty_spec = np.empty((60, 0), dtype=np.float32)
            empty_times = np.empty(0, dtype=np.float32)
            append_spectrogram_chunk(group, empty_spec, empty_times)

            # Shape should remain (60, 0)
            self.assertEqual(group[SPECTROGRAM_KEY].shape, (60, 0))
            self.assertEqual(group[TIMES_KEY].shape, (0,))

    def test_append_mismatched_freq_axis_raises(self) -> None:
        cfg = SpectrogramConfig()
        freqs_hz = np.linspace(30_000.0, 125_000.0, 100)
        sample_rate_hz = 250_000

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "mismatch.zarr"
            group = init_spectrogram_store(store_path, freqs_hz, sample_rate_hz, cfg)

            # Create chunk with wrong number of freq bins
            wrong_spec = np.random.randn(90, 50).astype(np.float32)
            times_s = np.linspace(0.0, 0.1, 50)

            with self.assertRaises(ValueError) as ctx:
                append_spectrogram_chunk(group, wrong_spec, times_s)
            self.assertIn("frequency axis does not match", str(ctx.exception))

    def test_append_mismatched_times_raises(self) -> None:
        cfg = SpectrogramConfig()
        freqs_hz = np.linspace(30_000.0, 125_000.0, 100)
        sample_rate_hz = 250_000

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "time_mismatch.zarr"
            group = init_spectrogram_store(store_path, freqs_hz, sample_rate_hz, cfg)

            spec_db = np.random.randn(100, 50).astype(np.float32)
            wrong_times = np.linspace(0.0, 0.1, 40)  # Wrong number of frames

            with self.assertRaises(ValueError) as ctx:
                append_spectrogram_chunk(group, spec_db, wrong_times)
            self.assertIn("times_s must match the number of frames", str(ctx.exception))

    def test_append_non_2d_spectrogram_raises(self) -> None:
        cfg = SpectrogramConfig()
        freqs_hz = np.linspace(30_000.0, 125_000.0, 100)
        sample_rate_hz = 250_000

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "non_2d.zarr"
            group = init_spectrogram_store(store_path, freqs_hz, sample_rate_hz, cfg)

            # 1D array
            bad_spec = np.random.randn(100).astype(np.float32)

            with self.assertRaises(ValueError) as ctx:
                append_spectrogram_chunk(group, bad_spec)
            self.assertIn("spec_db must be 2D", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
