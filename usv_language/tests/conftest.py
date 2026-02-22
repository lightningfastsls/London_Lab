"""Shared pytest fixtures for usv_language tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

# Bootstrap: add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import usv_language.src  # noqa: F401


@pytest.fixture
def synthetic_spectrogram() -> np.ndarray:
    """Create a synthetic spectrogram (n_freq=170, n_frames=1024)."""
    rng = np.random.RandomState(42)
    spec = rng.randn(170, 1024).astype(np.float32)
    # Simulate dB values roughly in [-100, 0] range
    spec = spec * 20 - 60
    return spec


@pytest.fixture
def synthetic_hdf5(synthetic_spectrogram, tmp_path):
    """Create a temporary HDF5 file with 3 synthetic spectrograms."""
    hdf5_path = tmp_path / "test_data.h5"
    with h5py.File(hdf5_path, "w") as f:
        f.attrs["sample_rate"] = 300_000
        f.attrs["n_fft"] = 512
        f.attrs["hop_length"] = 128
        f.attrs["freq_min_hz"] = 20_000
        f.attrs["freq_max_hz"] = 120_000
        f.attrs["eps"] = 1e-12
        f.attrs["n_files"] = 3
        f.attrs["total_frames"] = 0

        total_frames = 0
        for i in range(3):
            rng = np.random.RandomState(42 + i)
            n_frames = 512 + i * 256
            spec = rng.randn(170, n_frames).astype(np.float32) * 20 - 60

            grp = f.create_group(f"file_{i:03d}")
            grp.create_dataset("spectrogram", data=spec, dtype="float32")
            grp.attrs["n_frames"] = n_frames
            grp.attrs["sample_rate"] = 300_000
            total_frames += n_frames

        f.attrs["total_frames"] = total_frames

    return hdf5_path


@pytest.fixture
def synthetic_wav(tmp_path):
    """Create a temporary WAV file with a synthetic 60 kHz tone at 300 kHz SR."""
    import soundfile as sf

    sr = 300_000
    duration_s = 0.1
    n_samples = int(sr * duration_s)
    t = np.arange(n_samples) / sr
    signal = (
        0.01 * np.random.randn(n_samples).astype(np.float32)
        + 0.1 * np.sin(2 * np.pi * 60_000 * t).astype(np.float32)
    )

    wav_path = tmp_path / "test_tone.wav"
    sf.write(str(wav_path), signal, sr, subtype="FLOAT")
    return wav_path


@pytest.fixture
def wav_dir_with_files(tmp_path):
    """Create a directory with multiple synthetic WAV files."""
    import soundfile as sf

    sr = 300_000
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()

    for i in range(3):
        duration_s = 0.05 + i * 0.02
        n_samples = int(sr * duration_s)
        t = np.arange(n_samples) / sr
        freq = 50_000 + i * 10_000
        signal = (
            0.01 * np.random.randn(n_samples).astype(np.float32)
            + 0.1 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        )
        sf.write(str(wav_dir / f"test_{i:03d}.wav"), signal, sr, subtype="FLOAT")

    return wav_dir
