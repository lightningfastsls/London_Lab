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
class SonicConfig:
    """STFT config for sonic-range spectrogram (0-15 kHz).

    Runs a 1024-point FFT directly on the raw 300kHz signal (no decimation).
    This avoids IIR anti-aliasing filter smear from scipy.signal.decimate()
    AND keeps the analysis window short (3.4ms) for temporal sharpness.
    hop_length=128 matches the USV view for natural scroll sync.

    NOTE: freq_min_hz / freq_max_hz are INTENTIONALLY 0-30 kHz (NOT the
    corpus USV band from ``corpus.py``). This config drives the sonic-range
    preview pane, which shows audible-range energy for context — it is not
    part of the USV analysis pipeline. Do not "fix" these to 20-120 kHz.
    """

    sample_rate: int = 300_000     # Raw sample rate — no decimation
    n_fft: int = 1024              # ~293 Hz/bin at 300 kHz, 3.4ms window
    hop_length: int = 128          # Same as USV → matched column count
    window: str = "hann"
    freq_min_hz: int = 0
    freq_max_hz: int = 30_000      # Full sonic range 0-30 kHz


@dataclass
class AudioData:
    """Container for loaded audio and computed spectrogram."""

    audio: np.ndarray  # Shape: (samples,), float32 in [-1, 1]
    sample_rate: int  # Sample rate in Hz
    spectrogram_db: np.ndarray  # Shape: (freqs, times), dB-scaled
    frequencies: np.ndarray  # Frequency bins in Hz
    times: np.ndarray  # Time bins in seconds
    duration_s: float  # Total audio duration in seconds

    # Sonic-range spectrogram (0-30 kHz, separate STFT)
    sonic_spectrogram_db: np.ndarray | None = None  # Shape: (freqs, times)
    sonic_frequencies: np.ndarray | None = None      # Frequency bins in Hz
    sonic_times: np.ndarray | None = None            # Time bins (own axis)

    # Playback audio (48 kHz for sounddevice)
    playback_audio: np.ndarray | None = None
    playback_sr: int = 48_000


class AudioLoader:
    """Loads WAV files and computes spectrograms for detection.

    Uses ExtractionConfig parameters to ensure consistency with
    trained CNN model (n_fft=512, hop=128, freq=20-120kHz).
    """

    def __init__(
        self,
        config: ExtractionConfig | None = None,
        sonic_config: SonicConfig | None = None,
    ):
        """Initialize audio loader.

        Args:
            config: Extraction config for USV spectrogram (default: ExtractionConfig())
            sonic_config: Config for sonic-range spectrogram (default: SonicConfig())
        """
        self.config = config if config is not None else ExtractionConfig()
        self.sonic_config = sonic_config if sonic_config is not None else SonicConfig()

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

        # Compute sonic-range spectrogram (0-30 kHz)
        sonic_spec_db, sonic_freqs, sonic_times = self._compute_sonic_spectrogram(audio)

        # Compute playback audio (48 kHz)
        playback_audio = self._compute_playback_audio(audio)

        duration_s = len(audio) / sample_rate

        return AudioData(
            audio=audio,
            sample_rate=sample_rate,
            spectrogram_db=spec_db,
            frequencies=freqs_hz,
            times=times_s,
            duration_s=duration_s,
            sonic_spectrogram_db=sonic_spec_db,
            sonic_frequencies=sonic_freqs,
            sonic_times=sonic_times,
            playback_audio=playback_audio,
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

        # Compute STFT in dB with magnitude normalization
        # normalize_magnitude=True matches training pipeline (max dB = 0)
        eps = 1e-10  # Epsilon for log calculation
        spec_db = compute_stft_frames_db(
            frames, window, self.config.n_fft, band_mask, eps,
            normalize_magnitude=True
        )

        # Compute time bins (center of each frame)
        n_frames = frames.shape[0]
        times_s = (
            (np.arange(n_frames) * self.config.hop_length) + self.config.n_fft / 2.0
        ) / sample_rate

        return spec_db, freqs_hz[band_mask], times_s

    def _compute_sonic_spectrogram(
        self,
        samples: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute sonic-range spectrogram directly on raw 300kHz signal.

        Uses a 4096-point FFT with hop=128 (same as USV) to avoid
        decimation filter smear. Crops to 0-15 kHz (~205 frequency bins).

        Args:
            samples: Audio samples at 300kHz (float32)

        Returns:
            Tuple of (spec_db, freqs_hz, times_s) for 0-15 kHz range
        """
        sc = self.sonic_config

        # Build frequency bins for raw 300kHz signal
        freqs_hz = np.fft.rfftfreq(sc.n_fft, d=1.0 / sc.sample_rate)
        band_mask = (freqs_hz >= sc.freq_min_hz) & (freqs_hz <= sc.freq_max_hz)

        # Handle short audio
        if samples.size < sc.n_fft:
            return (
                np.empty((band_mask.sum(), 0)),
                freqs_hz[band_mask],
                np.empty(0),
            )

        # Extract frames and compute STFT on raw signal (no decimation)
        frames = extract_frames(samples, sc.n_fft, sc.hop_length)
        window = signal.get_window(sc.window, sc.n_fft, fftbins=True)
        eps = 1e-10

        # normalize_magnitude=False — display-only, not CNN input
        spec_db = compute_stft_frames_db(
            frames, window, sc.n_fft, band_mask, eps,
            normalize_magnitude=False,
        )

        # Compute time bins (center of each frame)
        n_frames = frames.shape[0]
        times_s = (
            (np.arange(n_frames) * sc.hop_length) + sc.n_fft / 2.0
        ) / sc.sample_rate

        return spec_db, freqs_hz[band_mask], times_s

    def _compute_playback_audio(self, samples: np.ndarray) -> np.ndarray:
        """Resample audio to 48kHz for sounddevice playback.

        Uses rational resampling: 300kHz * (8/50) = 48kHz.

        Args:
            samples: Audio at 300kHz

        Returns:
            Audio resampled to 48kHz, float32
        """
        return signal.resample_poly(samples, up=8, down=50).astype(np.float32)
