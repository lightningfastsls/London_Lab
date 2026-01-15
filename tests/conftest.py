"""Shared pytest fixtures for USV spectrogram tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.config import SpectrogramConfig
from usv_spectrogram.spectrogram import compute_spectrogram_db


@pytest.fixture
def sample_wav_path() -> Path:
    """Create a temporary WAV file with a synthetic USV-like signal.

    Returns a 250 kHz mono WAV with a 60 kHz tone embedded in noise,
    lasting 0.1 seconds (25000 samples).
    """
    sample_rate_hz = 250_000
    duration_s = 0.1
    n_samples = int(sample_rate_hz * duration_s)

    # Generate synthetic USV-like signal: noise + 60 kHz tone
    t = np.arange(n_samples) / sample_rate_hz
    noise = 0.01 * np.random.randn(n_samples).astype(np.float32)
    tone_freq_hz = 60_000.0
    tone = 0.1 * np.sin(2.0 * np.pi * tone_freq_hz * t).astype(np.float32)
    signal = noise + tone

    # Write to temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)

    sf.write(wav_path, signal, sample_rate_hz, subtype="FLOAT")

    yield wav_path

    # Cleanup
    if wav_path.exists():
        wav_path.unlink()


@pytest.fixture
def sample_config() -> SpectrogramConfig:
    """Return a default SpectrogramConfig for testing."""
    return SpectrogramConfig()


@pytest.fixture
def sample_spectrogram(sample_config: SpectrogramConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a pre-computed spectrogram tuple for testing.

    Returns (spec_db, freqs_hz, times_s) from a synthetic signal.
    """
    sample_rate_hz = 250_000
    duration_s = 0.05
    n_samples = int(sample_rate_hz * duration_s)

    # Generate synthetic signal with 70 kHz tone
    t = np.arange(n_samples) / sample_rate_hz
    noise = 0.01 * np.random.randn(n_samples).astype(np.float32)
    tone = 0.05 * np.sin(2.0 * np.pi * 70_000.0 * t).astype(np.float32)
    signal = noise + tone

    spec_db, freqs_hz, times_s = compute_spectrogram_db(
        signal, sample_rate_hz, sample_config
    )

    return spec_db, freqs_hz, times_s
