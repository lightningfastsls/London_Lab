"""WAV → spectrogram → HDF5 preprocessing pipeline.

Converts raw WAV recordings into continuous dB spectrograms stored in HDF5,
ready for chunking by the PyTorch Dataset.

Reuses existing STFT utilities from usv_spectrogram to avoid reimplementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from scipy import signal as scipy_signal

# Bootstrap path (importing this package triggers src/__init__.py)
import usv_language.src  # noqa: F401
from usv_spectrogram.io_wav import load_wav_mono
from usv_spectrogram._stft_core import extract_frames, compute_stft_frames_db

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class STFTConfig:
    """STFT parameters for the VQ-VAE data pipeline.

    Tuned for 300 kHz mouse USV recordings with 20-120 kHz band.
    """

    sample_rate: int = 300_000
    n_fft: int = 512
    hop_length: int = 128
    freq_min_hz: int = 20_000
    freq_max_hz: int = 120_000
    eps: float = 1e-12
    window: str = "hann"

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if self.hop_length <= 0 or self.hop_length >= self.n_fft:
            raise ValueError("hop_length must be positive and < n_fft")
        if self.freq_min_hz >= self.freq_max_hz:
            raise ValueError("freq_min_hz must be < freq_max_hz")
        if self.freq_max_hz > self.sample_rate // 2:
            raise ValueError("freq_max_hz exceeds Nyquist frequency")

    @property
    def n_freq_bins(self) -> int:
        """Number of frequency bins in the output spectrogram."""
        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)
        mask = (freqs >= self.freq_min_hz) & (freqs <= self.freq_max_hz)
        return int(mask.sum())

    @property
    def freq_resolution_hz(self) -> float:
        """Frequency resolution in Hz."""
        return self.sample_rate / self.n_fft

    @property
    def hop_duration_s(self) -> float:
        """Duration of one hop in seconds."""
        return self.hop_length / self.sample_rate


def compute_full_spectrogram(
    wav_path: str | Path,
    config: STFTConfig,
) -> np.ndarray:
    """Load a WAV file and compute a continuous dB spectrogram.

    Parameters
    ----------
    wav_path:
        Path to the WAV file.
    config:
        STFT configuration.

    Returns
    -------
    spec_db:
        dB spectrogram, shape (n_freq_bins, n_frames).
    """
    samples, sr = load_wav_mono(wav_path)

    if sr != config.sample_rate:
        logger.warning(
            "WAV sample rate %d != config %d for %s. Using file rate.",
            sr, config.sample_rate, wav_path,
        )

    window = scipy_signal.get_window(config.window, config.n_fft, fftbins=True)
    freqs = np.fft.rfftfreq(config.n_fft, d=1.0 / sr)
    band_mask = (freqs >= config.freq_min_hz) & (freqs <= config.freq_max_hz)

    frames = extract_frames(samples, config.n_fft, config.hop_length)
    if frames.shape[0] == 0:
        n_freq = int(band_mask.sum())
        return np.empty((n_freq, 0), dtype=np.float32)

    spec_db = compute_stft_frames_db(frames, window, config.n_fft, band_mask, config.eps)
    return spec_db.astype(np.float32)


def preprocess_single_wav(
    wav_path: str | Path,
    config: STFTConfig,
    norm_method: str = "none",
) -> dict:
    """Compute spectrogram for one WAV file.

    Parameters
    ----------
    wav_path:
        Path to WAV file.
    config:
        STFT configuration.
    norm_method:
        "none" returns raw dB values. Normalization is applied later globally.

    Returns
    -------
    dict with keys: "spectrogram" (ndarray), "wav_stem" (str),
    "sample_rate" (int), "n_frames" (int), "n_freq_bins" (int).
    """
    wav_path = Path(wav_path)
    spec_db = compute_full_spectrogram(wav_path, config)

    return {
        "spectrogram": spec_db,
        "wav_stem": wav_path.stem,
        "sample_rate": config.sample_rate,
        "n_frames": spec_db.shape[1],
        "n_freq_bins": spec_db.shape[0],
    }


def preprocess_all_wavs(
    wav_dir: str | Path,
    output_hdf5: str | Path,
    config: Optional[STFTConfig] = None,
    glob_pattern: str = "*.wav",
) -> dict:
    """Batch-process WAV files into a single HDF5 file.

    HDF5 structure:
        /<wav_stem>/spectrogram  — float32 array (n_freq_bins, n_frames)
        Attributes: sample_rate, n_fft, hop_length, freq_min_hz, freq_max_hz

    Parameters
    ----------
    wav_dir:
        Directory containing WAV files.
    output_hdf5:
        Output HDF5 file path.
    config:
        STFT configuration. Uses defaults if None.
    glob_pattern:
        Glob pattern for finding WAV files.

    Returns
    -------
    Summary dict with keys: "n_files", "total_frames", "n_freq_bins", "output_path".
    """
    if config is None:
        config = STFTConfig()

    wav_dir = Path(wav_dir)
    output_hdf5 = Path(output_hdf5)
    wav_files = sorted(wav_dir.glob(glob_pattern))

    if not wav_files:
        raise FileNotFoundError(f"No WAV files matching '{glob_pattern}' in {wav_dir}")

    total_frames = 0
    n_freq_bins = 0

    with h5py.File(output_hdf5, "w") as f:
        # Store config as root attributes
        f.attrs["sample_rate"] = config.sample_rate
        f.attrs["n_fft"] = config.n_fft
        f.attrs["hop_length"] = config.hop_length
        f.attrs["freq_min_hz"] = config.freq_min_hz
        f.attrs["freq_max_hz"] = config.freq_max_hz
        f.attrs["eps"] = config.eps

        for wav_path in wav_files:
            logger.info("Processing %s", wav_path.name)
            result = preprocess_single_wav(wav_path, config)

            grp = f.create_group(result["wav_stem"])
            grp.create_dataset(
                "spectrogram",
                data=result["spectrogram"],
                dtype="float32",
                compression="gzip",
                compression_opts=4,
            )
            grp.attrs["n_frames"] = result["n_frames"]
            grp.attrs["sample_rate"] = result["sample_rate"]

            total_frames += result["n_frames"]
            n_freq_bins = result["n_freq_bins"]

        f.attrs["total_frames"] = total_frames
        f.attrs["n_files"] = len(wav_files)

    return {
        "n_files": len(wav_files),
        "total_frames": total_frames,
        "n_freq_bins": n_freq_bins,
        "output_path": str(output_hdf5),
    }
