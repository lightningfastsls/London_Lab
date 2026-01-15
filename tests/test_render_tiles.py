"""Tests for PNG rendering utilities."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

# Set matplotlib backend before importing anything that uses it
import matplotlib
matplotlib.use("Agg")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.config import SpectrogramConfig
from usv_spectrogram.render_tiles import render_png, render_tiled_pages


class TestRenderTiles(unittest.TestCase):
    def test_render_png_creates_file(self) -> None:
        cfg = SpectrogramConfig()
        spec_db = np.random.randn(100, 200).astype(np.float32) * 10.0 - 60.0
        freqs_hz = np.linspace(30_000.0, 125_000.0, 100)
        times_s = np.linspace(0.0, 1.0, 200)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_spectrogram.png"

            render_png(spec_db, freqs_hz, times_s, output_path, cfg)

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_render_png_with_title(self) -> None:
        cfg = SpectrogramConfig()
        spec_db = np.random.randn(50, 100).astype(np.float32) * 10.0 - 60.0
        freqs_hz = np.linspace(30_000.0, 125_000.0, 50)
        times_s = np.linspace(0.0, 0.5, 100)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_with_title.png"

            render_png(spec_db, freqs_hz, times_s, output_path, cfg, title="Test USV")

            self.assertTrue(output_path.exists())

    def test_render_tiled_pages_creates_multiple_files(self) -> None:
        cfg = SpectrogramConfig(
            tile_duration_s=2.0,
            tiles_per_row=2,
            tiles_per_col=2,
        )
        # Create spectrogram covering 10 seconds (5 tiles at 2s each)
        n_freq_bins = 100
        n_time_bins = 2000
        spec_db = np.random.randn(n_freq_bins, n_time_bins).astype(np.float32) * 10.0 - 60.0
        freqs_hz = np.linspace(30_000.0, 125_000.0, n_freq_bins)
        times_s = np.linspace(0.0, 10.0, n_time_bins)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            base_name = "test_tiled"

            output_paths = render_tiled_pages(
                spec_db, freqs_hz, times_s, output_dir, base_name, cfg
            )

            # 5 tiles at 4 tiles per page = 2 pages
            expected_pages = 2
            self.assertEqual(len(output_paths), expected_pages)

            for path in output_paths:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)

    def test_render_tiled_pages_single_page(self) -> None:
        cfg = SpectrogramConfig(
            tile_duration_s=5.0,
            tiles_per_row=4,
            tiles_per_col=3,
        )
        # Create short spectrogram (1 second, fits in 1 tile)
        spec_db = np.random.randn(80, 200).astype(np.float32) * 10.0 - 60.0
        freqs_hz = np.linspace(30_000.0, 125_000.0, 80)
        times_s = np.linspace(0.0, 1.0, 200)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            base_name = "short"

            output_paths = render_tiled_pages(
                spec_db, freqs_hz, times_s, output_dir, base_name, cfg
            )

            self.assertEqual(len(output_paths), 1)
            self.assertTrue(output_paths[0].name.startswith("short_page"))

    def test_render_tiled_pages_with_title(self) -> None:
        cfg = SpectrogramConfig(tile_duration_s=1.0)
        spec_db = np.random.randn(50, 250).astype(np.float32) * 10.0 - 60.0
        freqs_hz = np.linspace(30_000.0, 125_000.0, 50)
        times_s = np.linspace(0.0, 2.0, 250)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            output_paths = render_tiled_pages(
                spec_db, freqs_hz, times_s, output_dir, "titled", cfg, title="Test Title"
            )

            self.assertGreater(len(output_paths), 0)

    def test_output_is_valid_png(self) -> None:
        # Verify PNG header signature
        cfg = SpectrogramConfig()
        spec_db = np.random.randn(50, 100).astype(np.float32) * 10.0 - 60.0
        freqs_hz = np.linspace(30_000.0, 125_000.0, 50)
        times_s = np.linspace(0.0, 0.5, 100)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "valid.png"

            render_png(spec_db, freqs_hz, times_s, output_path, cfg)

            # Check PNG signature (first 8 bytes)
            with open(output_path, "rb") as f:
                png_signature = f.read(8)

            expected_signature = b"\x89PNG\r\n\x1a\n"
            self.assertEqual(png_signature, expected_signature)

    def test_render_tiled_pages_empty_times_raises(self) -> None:
        cfg = SpectrogramConfig()
        spec_db = np.empty((50, 0), dtype=np.float32)
        freqs_hz = np.linspace(30_000.0, 125_000.0, 50)
        times_s = np.empty(0, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError) as ctx:
                render_tiled_pages(
                    spec_db, freqs_hz, times_s, Path(tmpdir), "empty", cfg
                )
            self.assertIn("times_s is empty", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
