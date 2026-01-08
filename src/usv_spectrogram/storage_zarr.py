"""Zarr storage helpers for incremental spectrogram output."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import zarr

from .config import SpectrogramConfig

SPECTROGRAM_KEY = "spectrogram_db"
FREQS_KEY = "freqs_hz"
TIMES_KEY = "times_s"
SCALE_DB_PER_LSB = 0.1


def init_spectrogram_store(
    path: str | Path,
    freqs_hz: np.ndarray,
    sample_rate_hz: int,
    cfg: SpectrogramConfig,
    overwrite: bool = True,
) -> zarr.Group:
    """Initialize a Zarr group for incremental spectrogram writes."""
    mode = "w" if overwrite else "a"
    group = zarr.open_group(str(path), mode=mode)

    group.attrs.update(
        {
            "sample_rate_hz": int(sample_rate_hz),
            "window_length": int(cfg.window_length),
            "hop_length": int(cfg.hop_length(sample_rate_hz)),
            "n_fft": int(cfg.n_fft()),
            "f_min_hz": float(cfg.f_min_hz),
            "f_max_hz": float(cfg.f_max_hz),
            "scale_db_per_lsb": float(SCALE_DB_PER_LSB),
            "units": "dB",
        }
    )

    group.create_dataset(
        FREQS_KEY,
        shape=freqs_hz.shape,
        data=freqs_hz.astype(np.float32, copy=False),
        overwrite=True,
    )
    group.create_dataset(
        TIMES_KEY,
        shape=(0,),
        chunks=(cfg.zarr_time_chunk_frames,),
        dtype="f8",
        overwrite=True,
    )
    group.create_dataset(
        SPECTROGRAM_KEY,
        shape=(freqs_hz.size, 0),
        chunks=(freqs_hz.size, cfg.zarr_time_chunk_frames),
        dtype="i2",
        overwrite=True,
    )
    return group


def append_spectrogram_chunk(
    group: zarr.Group,
    spec_db: np.ndarray,
    times_s: Optional[np.ndarray] = None,
) -> None:
    """Append a spectrogram chunk (in dB) to the Zarr store."""
    if spec_db.ndim != 2:
        raise ValueError("spec_db must be 2D (freqs, frames).")

    store = group[SPECTROGRAM_KEY]
    if spec_db.shape[0] != store.shape[0]:
        raise ValueError("spec_db frequency axis does not match store.")

    frame_count = spec_db.shape[1]
    if frame_count == 0:
        return

    start = store.shape[1]
    store.resize((store.shape[0], start + frame_count))
    store[:, start : start + frame_count] = _db_to_int16(spec_db)

    if times_s is not None:
        if times_s.shape[0] != frame_count:
            raise ValueError("times_s must match the number of frames.")
        times_store = group[TIMES_KEY]
        times_store.resize(start + frame_count)
        times_store[start : start + frame_count] = times_s


def _db_to_int16(spec_db: np.ndarray) -> np.ndarray:
    scaled = np.round(spec_db / SCALE_DB_PER_LSB)
    clipped = np.clip(scaled, np.iinfo(np.int16).min, np.iinfo(np.int16).max)
    return clipped.astype(np.int16, copy=False)
