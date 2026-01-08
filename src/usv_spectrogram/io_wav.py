"""WAV loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf


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
