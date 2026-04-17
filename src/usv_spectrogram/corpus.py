"""Corpus constants — physical facts shared across all USV analysis modules.

Single source of truth for sample rate, USV frequency band, and STFT
parameters. Imported by ``SpectrogramConfig``, ``DetectionConfig``,
``ExtractionConfig`` (via drift assertion), ``AnalysisConfig``, and any
analysis script that needs these values. No module should redeclare these.

Values chosen 2026-04-17:
    - Sample rate: 300 kHz (matches LMT rig; Nyquist = 150 kHz covers the
      mouse USV band with headroom).
    - USV band: 20–120 kHz (aligned with CNN training; see constraint below).
    - STFT: n_fft=512, hop=128 (matches CLAUDE.md ADR-002 — ~586 Hz freq
      resolution, ~1.7 ms time resolution at 300 kHz, 75 % overlap).

CRITICAL CONSTRAINT — values are LOCKED to the production CNN.

    The model at ``models/hard_neg_retrain/best_model.pt`` was trained on
    256-pixel-tall spectrograms covering exactly 20–120 kHz. Changing
    ``USV_FREQ_MIN_HZ`` or ``USV_FREQ_MAX_HZ`` here without retraining the
    CNN produces silently-wrong inference (same pixel grid, different
    Hz-per-pixel, features shifted to the wrong rows). If the corpus band
    ever needs to change, the CNN must be retrained first — and then
    ``ExtractionConfig`` updated to match, in that order.
"""

from __future__ import annotations

from typing import Final

SAMPLE_RATE_HZ: Final[int] = 300_000

USV_FREQ_MIN_HZ: Final[int] = 20_000
USV_FREQ_MAX_HZ: Final[int] = 120_000

STFT_N_FFT: Final[int] = 512
STFT_HOP: Final[int] = 128


def nyquist_hz() -> int:
    return SAMPLE_RATE_HZ // 2


def stft_freq_resolution_hz() -> float:
    return SAMPLE_RATE_HZ / STFT_N_FFT


def stft_time_resolution_ms() -> float:
    return (STFT_N_FFT / SAMPLE_RATE_HZ) * 1000.0


def stft_hop_ms() -> float:
    return (STFT_HOP / SAMPLE_RATE_HZ) * 1000.0
