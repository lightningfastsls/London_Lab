"""Bout-level spectrogram computation.

Computes STFT spectrograms for bout audio segments using the shared
_stft_core functions, ensuring identical DSP parameters to the CNN
detection pipeline.

References:
    ADR-001 (sample rate: 300 kHz)
    ADR-002 (STFT parameters: n_fft=512, hop=128, 20-120 kHz)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import get_window

# Bootstrap: ensure project root is on sys.path for _stft_core import
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.usv_spectrogram._stft_core import compute_stft_frames_db, extract_frames


@dataclass(frozen=True)
class BoutSpectrogramConfig:
    """Configuration for bout spectrogram computation.

    All parameters match the CNN detection pipeline (ADR-002).
    """

    sample_rate: int = 300_000
    n_fft: int = 512
    hop_length: int = 128
    freq_min_hz: int = 20_000
    freq_max_hz: int = 120_000
    window: str = "hann"
    eps: float = 1e-10

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.n_fft <= 0:
            raise ValueError(f"n_fft must be positive, got {self.n_fft}")
        if self.hop_length <= 0:
            raise ValueError(f"hop_length must be positive, got {self.hop_length}")
        if self.freq_min_hz >= self.freq_max_hz:
            raise ValueError(
                f"freq_min_hz ({self.freq_min_hz}) must be less than "
                f"freq_max_hz ({self.freq_max_hz})"
            )
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")

    def get_freq_bins(self) -> np.ndarray:
        """Return frequency values for each rfft bin."""
        return np.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)

    def get_band_mask(self) -> np.ndarray:
        """Return boolean mask selecting bins in [freq_min_hz, freq_max_hz]."""
        freqs = self.get_freq_bins()
        return (freqs >= self.freq_min_hz) & (freqs <= self.freq_max_hz)

    def get_n_freq_bins(self) -> int:
        """Return number of frequency bins in the selected band."""
        return int(self.get_band_mask().sum())


def compute_bout_spectrogram(
    wav_path: str | Path,
    start_time_s: float,
    end_time_s: float,
    config: BoutSpectrogramConfig | None = None,
) -> np.ndarray:
    """Compute a dB spectrogram for a bout audio segment.

    Parameters
    ----------
    wav_path:
        Path to the WAV file.
    start_time_s:
        Bout start time in seconds.
    end_time_s:
        Bout end time in seconds.
    config:
        Spectrogram configuration. Uses defaults if None.

    Returns
    -------
    spec_db:
        Spectrogram in dB, shape (n_freq_bins, T) where n_freq_bins is
        typically 170 (20-120 kHz at sr=300k, n_fft=512).
    """
    if config is None:
        config = BoutSpectrogramConfig()

    # Load audio segment
    audio = _load_audio_segment(wav_path, start_time_s, end_time_s, config.sample_rate)

    if len(audio) < config.n_fft:
        # Audio too short for even one frame — return empty spectrogram
        n_freq = config.get_n_freq_bins()
        return np.empty((n_freq, 0), dtype=np.float32)

    # Extract frames using shared _stft_core function
    frames = extract_frames(audio, config.n_fft, config.hop_length)

    if frames.shape[0] == 0:
        n_freq = config.get_n_freq_bins()
        return np.empty((n_freq, 0), dtype=np.float32)

    # Compute windowed STFT in dB
    window = get_window(config.window, config.n_fft).astype(np.float32)
    band_mask = config.get_band_mask()

    spec_db = compute_stft_frames_db(
        frames, window, config.n_fft, band_mask, config.eps,
    )

    return spec_db.astype(np.float32)


def _load_audio_segment(
    wav_path: str | Path,
    start_time_s: float,
    end_time_s: float,
    sample_rate: int,
) -> np.ndarray:
    """Load a segment of audio from a WAV file.

    Uses soundfile for efficient seeking without loading the entire file.
    """
    wav_path = Path(wav_path)

    with sf.SoundFile(str(wav_path)) as wav:
        file_sr = wav.samplerate
        if file_sr != sample_rate:
            raise ValueError(
                f"WAV sample rate {file_sr} != expected {sample_rate} "
                f"for {wav_path}"
            )

        total_frames = wav.frames
        start_sample = max(0, int(round(start_time_s * sample_rate)))
        end_sample = min(total_frames, int(round(end_time_s * sample_rate)))

        if end_sample <= start_sample:
            return np.empty(0, dtype=np.float32)

        wav.seek(start_sample)
        n_samples = end_sample - start_sample
        audio = wav.read(n_samples, dtype="float32", always_2d=True)

    if audio.size == 0:
        return np.empty(0, dtype=np.float32)

    # Convert to mono if needed
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]

    return audio
