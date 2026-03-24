"""Tests for SpectrogramConfig."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.config import SpectrogramConfig


class TestConfig(unittest.TestCase):
    def test_default_config_creation(self) -> None:
        cfg = SpectrogramConfig()
        self.assertEqual(cfg.expected_sample_rate_hz, 250_000)
        self.assertEqual(cfg.window_length, 2048)
        self.assertEqual(cfg.zero_padding_factor, 2)
        self.assertEqual(cfg.hop_ms, 0.5)
        self.assertEqual(cfg.f_min_hz, 30_000.0)
        self.assertEqual(cfg.f_max_hz, 125_000.0)
        self.assertTrue(cfg.enforce_sample_rate)

    def test_n_fft_returns_correct_value(self) -> None:
        cfg = SpectrogramConfig(window_length=1024, zero_padding_factor=2)
        self.assertEqual(cfg.n_fft(), 2048)

        cfg = SpectrogramConfig(window_length=2048, zero_padding_factor=4)
        self.assertEqual(cfg.n_fft(), 8192)

    def test_hop_length_with_different_sample_rates(self) -> None:
        cfg = SpectrogramConfig(hop_ms=0.5)

        # At 250 kHz, 0.5 ms = 125 samples
        self.assertEqual(cfg.hop_length(250_000), 125)

        # At 100 kHz, 0.5 ms = 50 samples
        self.assertEqual(cfg.hop_length(100_000), 50)

        # At 1 kHz, 0.5 ms = 0.5 samples, rounds to 1 (minimum)
        self.assertEqual(cfg.hop_length(1_000), 1)

    def test_post_init_raises_on_invalid_window_length(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            SpectrogramConfig(window_length=0)
        self.assertIn("window_length must be positive", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            SpectrogramConfig(window_length=-100)
        self.assertIn("window_length must be positive", str(ctx.exception))

    def test_post_init_raises_on_invalid_zero_padding_factor(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            SpectrogramConfig(zero_padding_factor=0)
        self.assertIn("zero_padding_factor must be >= 1", str(ctx.exception))

    def test_post_init_raises_on_invalid_frequency_range(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            SpectrogramConfig(f_min_hz=100_000.0, f_max_hz=50_000.0)
        self.assertIn("f_min_hz must be < f_max_hz", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            SpectrogramConfig(f_min_hz=60_000.0, f_max_hz=60_000.0)
        self.assertIn("f_min_hz must be < f_max_hz", str(ctx.exception))

    def test_post_init_raises_on_invalid_hop_ms(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            SpectrogramConfig(hop_ms=0.0)
        self.assertIn("hop_ms must be positive", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            SpectrogramConfig(hop_ms=-1.0)
        self.assertIn("hop_ms must be positive", str(ctx.exception))

    def test_post_init_raises_on_invalid_eps(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            SpectrogramConfig(eps=0.0)
        self.assertIn("eps must be positive", str(ctx.exception))

    def test_post_init_raises_on_invalid_stream_block_size(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            SpectrogramConfig(stream_block_size_samples=0)
        self.assertIn("stream_block_size_samples must be positive", str(ctx.exception))

    def test_post_init_raises_on_invalid_zarr_chunk_frames(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            SpectrogramConfig(zarr_time_chunk_frames=-1)
        self.assertIn("zarr_time_chunk_frames must be positive", str(ctx.exception))

    def test_frozen_dataclass_cannot_modify_after_creation(self) -> None:
        cfg = SpectrogramConfig()
        with self.assertRaises(Exception):
            cfg.window_length = 1024  # type: ignore


if __name__ == "__main__":
    unittest.main()
