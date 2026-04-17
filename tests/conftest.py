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
from usv_spectrogram.detection.config import DetectionConfig


@pytest.fixture
def sample_wav_path() -> Path:
    """Create a temporary WAV file with a synthetic USV-like signal.

    Returns a 250 kHz mono WAV with a 60 kHz tone embedded in noise,
    lasting 0.1 seconds (25000 samples).
    """
    sample_rate_hz = 300_000
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
    sample_rate_hz = 300_000
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


# Detection-specific fixtures


@pytest.fixture
def detection_config() -> DetectionConfig:
    """Return a default DetectionConfig for testing."""
    return DetectionConfig()


@pytest.fixture
def create_tone_wav():
    """Factory fixture to create WAV files with synthetic tones.

    Returns a function that creates a temporary WAV file with specified
    frequency, duration, and amplitude. Yields the path and cleans up after.
    """
    created_paths = []

    def _create(
        freq_hz: float,
        duration_ms: float,
        amplitude: float = 0.5,
        sample_rate: int = 300_000,
        noise_level: float = 0.01,
        start_offset_ms: float = 50.0,
    ) -> Path:
        """Create a WAV file with a tone at specified frequency.

        Parameters
        ----------
        freq_hz: Frequency of the tone in Hz
        duration_ms: Duration of the tone in milliseconds
        amplitude: Amplitude of tone (0-1)
        sample_rate: Sample rate in Hz
        noise_level: Background noise amplitude
        start_offset_ms: Time before tone starts (to avoid edge effects)
        """
        # Total duration includes padding on both sides
        total_duration_ms = start_offset_ms + duration_ms + 50.0
        n_samples = int(sample_rate * total_duration_ms / 1000.0)

        # Create time array
        t = np.arange(n_samples) / sample_rate

        # Background noise
        signal = noise_level * np.random.randn(n_samples).astype(np.float32)

        # Add tone in the middle section
        start_sample = int(start_offset_ms / 1000.0 * sample_rate)
        tone_samples = int(duration_ms / 1000.0 * sample_rate)
        end_sample = start_sample + tone_samples

        tone_t = t[start_sample:end_sample]
        tone = amplitude * np.sin(2.0 * np.pi * freq_hz * tone_t).astype(np.float32)
        signal[start_sample:end_sample] += tone

        # Write to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)

        sf.write(wav_path, signal, sample_rate, subtype="FLOAT")
        created_paths.append(wav_path)
        return wav_path

    yield _create

    # Cleanup all created files
    for p in created_paths:
        if p.exists():
            p.unlink()


@pytest.fixture
def create_multi_tone_wav():
    """Factory fixture to create WAV files with multiple tones.

    Returns a function that creates a WAV file with multiple tones
    at different times.
    """
    created_paths = []

    def _create(
        tones: list[dict],
        sample_rate: int = 300_000,
        total_duration_ms: float = 500.0,
        noise_level: float = 0.01,
    ) -> Path:
        """Create WAV with multiple tones.

        Parameters
        ----------
        tones: List of dicts with keys:
            - freq_hz: Frequency in Hz
            - start_ms: Start time in ms
            - duration_ms: Duration in ms
            - amplitude: Optional amplitude (default 0.5)
        sample_rate: Sample rate in Hz
        total_duration_ms: Total file duration
        noise_level: Background noise level
        """
        n_samples = int(sample_rate * total_duration_ms / 1000.0)
        t = np.arange(n_samples) / sample_rate

        # Background noise
        signal = noise_level * np.random.randn(n_samples).astype(np.float32)

        # Add each tone
        for tone_spec in tones:
            freq_hz = tone_spec["freq_hz"]
            start_ms = tone_spec["start_ms"]
            duration_ms = tone_spec["duration_ms"]
            amplitude = tone_spec.get("amplitude", 0.5)

            start_sample = int(start_ms / 1000.0 * sample_rate)
            end_sample = start_sample + int(duration_ms / 1000.0 * sample_rate)
            end_sample = min(end_sample, n_samples)

            if start_sample < n_samples:
                tone_t = t[start_sample:end_sample]
                tone = amplitude * np.sin(2.0 * np.pi * freq_hz * tone_t).astype(np.float32)
                signal[start_sample:end_sample] += tone

        # Write to file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = Path(tmp.name)

        sf.write(wav_path, signal, sample_rate, subtype="FLOAT")
        created_paths.append(wav_path)
        return wav_path

    yield _create

    # Cleanup
    for p in created_paths:
        if p.exists():
            p.unlink()
