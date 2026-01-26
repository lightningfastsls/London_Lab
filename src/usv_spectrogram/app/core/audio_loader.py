"""Audio loading and spectrogram computation for detection app.

This module wraps existing io_wav and spectrogram functionality with app-specific
configuration. MUST use extraction_config.py parameters for consistency with
trained CNN model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from ...detection.extraction_config import ExtractionConfig
from ...io_wav import load_wav_mono
from ..._stft_core import compute_stft_frames_db, extract_frames
from scipy import signal


@dataclass
class AudioData:
    """Container for loaded audio and computed spectrogram."""

    audio: np.ndarray  # Shape: (samples,), float32 in [-1, 1]
    sample_rate: int  # Sample rate in Hz
    spectrogram_db: np.ndarray  # Shape: (freqs, times), dB-scaled
    frequencies: np.ndarray  # Frequency bins in Hz
    times: np.ndarray  # Time bins in seconds
    duration_s: float  # Total audio duration in seconds


class AudioLoader:
    """Loads WAV files and computes spectrograms for detection.

    Uses ExtractionConfig parameters to ensure consistency with
    trained CNN model (n_fft=512, hop=128, freq=20-120kHz).
    """

    def __init__(self, config: ExtractionConfig | None = None):
        """Initialize audio loader.

        Args:
            config: Extraction config (default: ExtractionConfig())
        """
        self.config = config if config is not None else ExtractionConfig()

    def load(self, wav_path: str | Path) -> AudioData:
        """Load WAV file and compute spectrogram.

        Args:
            wav_path: Path to WAV file

        Returns:
            AudioData containing audio samples and spectrogram

        Raises:
            FileNotFoundError: If WAV file doesn't exist
            ValueError: If sample rate doesn't match expected
        """
        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise FileNotFoundError(f"WAV file not found: {wav_path}")

        # Load audio
        audio, sample_rate = load_wav_mono(wav_path)

        # Validate sample rate
        if sample_rate != self.config.sample_rate:
            raise ValueError(
                f"Expected sample rate {self.config.sample_rate} Hz, "
                f"got {sample_rate} Hz. File: {wav_path}"
            )

        # Compute spectrogram using extraction config parameters
        spec_db, freqs_hz, times_s = self._compute_spectrogram(audio, sample_rate)

        duration_s = len(audio) / sample_rate

        return AudioData(
            audio=audio,
            sample_rate=sample_rate,
            spectrogram_db=spec_db,
            frequencies=freqs_hz,
            times=times_s,
            duration_s=duration_s
        )

    def _compute_spectrogram(
        self,
        samples: np.ndarray,
        sample_rate: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute dB-scaled spectrogram using extraction config parameters.

        Args:
            samples: Audio samples (float32)
            sample_rate: Sample rate in Hz

        Returns:
            Tuple of (spec_db, freqs_hz, times_s)
        """
        # Build frequency bins
        freqs_hz = np.fft.rfftfreq(self.config.n_fft, d=1.0 / sample_rate)

        # Apply frequency band filter
        band_mask = (
            (freqs_hz >= self.config.freq_min_hz) &
            (freqs_hz <= self.config.freq_max_hz)
        )

        # Handle short audio
        if samples.size < self.config.n_fft:
            return (
                np.empty((band_mask.sum(), 0)),
                freqs_hz[band_mask],
                np.empty(0)
            )

        # Extract frames using specified hop length
        frames = extract_frames(samples, self.config.n_fft, self.config.hop_length)

        # Get window
        window = signal.get_window(self.config.window, self.config.n_fft, fftbins=True)

        # Compute STFT in dB
        eps = 1e-10  # Epsilon for log calculation
        spec_db = compute_stft_frames_db(frames, window, self.config.n_fft, band_mask, eps)

        # Compute time bins (center of each frame)
        n_frames = frames.shape[0]
        times_s = (
            (np.arange(n_frames) * self.config.hop_length) + self.config.n_fft / 2.0
        ) / sample_rate

        return spec_db, freqs_hz[band_mask], times_s
