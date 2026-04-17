"""Tests for the corpus constants module.

The point of this module is that it is the *single source of truth* for
physical facts shared across every USV analysis pipeline. These tests
guard against two classes of regression:

1. Accidental value drift — someone "tidying up" the numbers.
2. CNN pixel-grid divergence — ``ExtractionConfig`` defaults must stay
   equal to corpus values because the CNN is frozen on that grid.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram import corpus
from usv_spectrogram.detection.extraction_config import ExtractionConfig


class TestCorpusConstants(unittest.TestCase):
    def test_sample_rate_is_300_khz(self) -> None:
        self.assertEqual(corpus.SAMPLE_RATE_HZ, 300_000)

    def test_usv_band_is_20_to_120_khz(self) -> None:
        self.assertEqual(corpus.USV_FREQ_MIN_HZ, 20_000)
        self.assertEqual(corpus.USV_FREQ_MAX_HZ, 120_000)
        self.assertLess(corpus.USV_FREQ_MIN_HZ, corpus.USV_FREQ_MAX_HZ)

    def test_stft_params_match_adr002(self) -> None:
        self.assertEqual(corpus.STFT_N_FFT, 512)
        self.assertEqual(corpus.STFT_HOP, 128)

    def test_nyquist_covers_usv_band(self) -> None:
        self.assertEqual(corpus.nyquist_hz(), 150_000)
        self.assertGreater(corpus.nyquist_hz(), corpus.USV_FREQ_MAX_HZ)

    def test_stft_freq_resolution(self) -> None:
        self.assertAlmostEqual(corpus.stft_freq_resolution_hz(), 585.9375, places=4)

    def test_stft_time_resolution(self) -> None:
        self.assertAlmostEqual(corpus.stft_time_resolution_ms(), 1.70666, places=4)

    def test_stft_hop_ms(self) -> None:
        self.assertAlmostEqual(corpus.stft_hop_ms(), 0.42666, places=4)

    def test_75_percent_stft_overlap(self) -> None:
        overlap = 1.0 - corpus.STFT_HOP / corpus.STFT_N_FFT
        self.assertAlmostEqual(overlap, 0.75, places=6)


class TestExtractionConfigDriftAssertion(unittest.TestCase):
    def test_extraction_defaults_equal_corpus_values(self) -> None:
        cfg = ExtractionConfig()
        self.assertEqual(cfg.sample_rate, corpus.SAMPLE_RATE_HZ)
        self.assertEqual(cfg.n_fft, corpus.STFT_N_FFT)
        self.assertEqual(cfg.hop_length, corpus.STFT_HOP)
        self.assertEqual(cfg.freq_min_hz, corpus.USV_FREQ_MIN_HZ)
        self.assertEqual(cfg.freq_max_hz, corpus.USV_FREQ_MAX_HZ)


class TestCorpusUsedByDownstream(unittest.TestCase):
    """Smoke-test: every downstream config class actually imports the values."""

    def test_spectrogram_config_uses_corpus(self) -> None:
        from usv_spectrogram.config import SpectrogramConfig
        cfg = SpectrogramConfig()
        self.assertEqual(cfg.expected_sample_rate_hz, corpus.SAMPLE_RATE_HZ)
        self.assertEqual(cfg.f_min_hz, float(corpus.USV_FREQ_MIN_HZ))
        self.assertEqual(cfg.f_max_hz, float(corpus.USV_FREQ_MAX_HZ))

    def test_detection_config_uses_corpus(self) -> None:
        from usv_spectrogram.detection.config import DetectionConfig
        cfg = DetectionConfig()
        self.assertEqual(cfg.sample_rate, corpus.SAMPLE_RATE_HZ)
        self.assertEqual(cfg.n_fft, corpus.STFT_N_FFT)
        self.assertEqual(cfg.hop_length, corpus.STFT_HOP)
        self.assertEqual(cfg.freq_min_hz, corpus.USV_FREQ_MIN_HZ)
        self.assertEqual(cfg.freq_max_hz, corpus.USV_FREQ_MAX_HZ)

    def test_analysis_config_uses_corpus(self) -> None:
        # usv_language lives at repo root, not under src/
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from usv_language.analysis.config import AnalysisConfig
        cfg = AnalysisConfig()
        self.assertEqual(cfg.freq_min_hz, corpus.USV_FREQ_MIN_HZ)
        self.assertEqual(cfg.freq_max_hz, corpus.USV_FREQ_MAX_HZ)


if __name__ == "__main__":
    unittest.main()
