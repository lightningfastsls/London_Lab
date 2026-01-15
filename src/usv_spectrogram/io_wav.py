"""WAV loading helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf


def get_default_wav_dir() -> Path:
    """Return the default WAV directory from env var or repo fallback.

    Checks USV_WAV_DIR environment variable first, falls back to
    <repo_root>/5970 USV if not set.
    """
    env_path = os.environ.get("USV_WAV_DIR")
    if env_path:
        return Path(env_path)
    # Fallback to repo-relative path
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "5970 USV"


def load_wav_mono(path: str | Path) -> Tuple[np.ndarray, int]:
    """Load a WAV file and return mono samples and sample rate.

    Parameters
    ----------
    path:
        Path to the WAV file.

    Returns
    -------
    samples:
        Mono float32 samples in range [-1, 1].
    sample_rate_hz:
        Sample rate reported by the WAV file.
    """
    samples, sample_rate_hz = sf.read(str(path), dtype="float32", always_2d=True)
    if samples.shape[1] > 1:
        samples = samples.mean(axis=1)
    else:
        samples = samples[:, 0]
    return samples, int(sample_rate_hz)


def load_wav_segment_mono(
    path: str | Path,
    start_s: float,
    duration_s: float,
) -> Tuple[np.ndarray, int]:
    """Load a segment of a WAV file and return mono samples and sample rate.

    Parameters
    ----------
    path:
        Path to the WAV file.
    start_s:
        Segment start time in seconds.
    duration_s:
        Segment duration in seconds.

    Returns
    -------
    samples:
        Mono float32 samples in range [-1, 1].
    sample_rate_hz:
        Sample rate reported by the WAV file.
    """
    if start_s < 0:
        raise ValueError("start_s must be non-negative.")

    with sf.SoundFile(str(path)) as wav:
        sample_rate_hz = int(wav.samplerate)
        if duration_s <= 0:
            return np.empty(0, dtype=np.float32), sample_rate_hz
        start_frame = int(round(start_s * sample_rate_hz))
        if start_frame < 0:
            start_frame = 0
        wav.seek(start_frame)
        frames_to_read = int(round(duration_s * sample_rate_hz))
        if frames_to_read <= 0:
            return np.empty(0, dtype=np.float32), sample_rate_hz
        samples = wav.read(frames_to_read, dtype="float32", always_2d=True)

    if samples.size == 0:
        return np.empty(0, dtype=np.float32), sample_rate_hz
    if samples.shape[1] > 1:
        samples = samples.mean(axis=1)
    else:
        samples = samples[:, 0]
    return samples, sample_rate_hz
