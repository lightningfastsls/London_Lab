"""PyTorch Dataset for spectrogram chunks from HDF5 storage.

Provides lazy-loading of fixed-length spectrogram chunks with configurable
overlap and silence filtering. Chunks are returned as (seq_len, n_freq)
tensors suitable for the column-wise encoder.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SpectrogramChunkDataset(Dataset):
    """Loads fixed-length spectrogram chunks from an HDF5 file.

    Each chunk is a (chunk_length, n_freq) slice of a continuous spectrogram.
    Silence chunks (low mean energy) are optionally filtered out.

    Parameters
    ----------
    hdf5_path:
        Path to preprocessed HDF5 file.
    chunk_length:
        Number of time columns per chunk (~109ms at hop=128, sr=300k).
    overlap_fraction:
        Fraction of overlap between consecutive chunks (0.0 to <1.0).
    silence_threshold:
        Skip chunks where mean absolute value is below this. Set to 0 to disable.
    file_stems:
        If provided, only use these files from the HDF5.
    norm_params:
        Optional (min_val, max_val) tuple for min-max normalization.
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        chunk_length: int = 256,
        overlap_fraction: float = 0.5,
        silence_threshold: float = 0.05,
        file_stems: Optional[List[str]] = None,
        norm_params: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.hdf5_path = str(hdf5_path)
        self.chunk_length = chunk_length
        self.overlap_fraction = overlap_fraction
        self.silence_threshold = silence_threshold
        self.norm_params = norm_params

        if not 0.0 <= overlap_fraction < 1.0:
            raise ValueError("overlap_fraction must be in [0.0, 1.0)")

        self.stride = max(1, int(chunk_length * (1.0 - overlap_fraction)))

        # Build chunk index: list of (file_stem, start_col)
        self._chunks: List[Tuple[str, int]] = []
        self._build_index(file_stems)

    def _build_index(self, file_stems: Optional[List[str]]) -> None:
        """Scan HDF5 and build chunk index, filtering silence."""
        with h5py.File(self.hdf5_path, "r") as f:
            stems = file_stems if file_stems is not None else list(f.keys())

            for stem in stems:
                if stem not in f or "spectrogram" not in f[stem]:
                    logger.warning("Skipping missing stem: %s", stem)
                    continue

                n_frames = f[stem]["spectrogram"].shape[1]
                n_chunks = max(0, (n_frames - self.chunk_length) // self.stride + 1)

                for i in range(n_chunks):
                    start_col = i * self.stride

                    # Silence filtering
                    if self.silence_threshold > 0:
                        chunk_data = f[stem]["spectrogram"][:, start_col:start_col + self.chunk_length]
                        if np.mean(np.abs(chunk_data)) < self.silence_threshold:
                            continue

                    self._chunks.append((stem, start_col))

        logger.info(
            "Built dataset: %d chunks from %d files",
            len(self._chunks),
            len(set(s for s, _ in self._chunks)),
        )

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        stem, start_col = self._chunks[idx]

        with h5py.File(self.hdf5_path, "r") as f:
            # Shape: (n_freq, chunk_length)
            chunk = f[stem]["spectrogram"][:, start_col:start_col + self.chunk_length]

        chunk = chunk.astype(np.float32)

        # Apply min-max normalization if params provided
        if self.norm_params is not None:
            min_val, max_val = self.norm_params
            denom = max_val - min_val
            if denom > 0:
                chunk = (chunk - min_val) / denom

        # Transpose to (seq_len, n_freq) for column-wise processing
        chunk = chunk.T

        return {
            "spectrogram": torch.from_numpy(chunk),
            "file_name": stem,
            "start_col": start_col,
        }


def create_splits(
    hdf5_path: str | Path,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Split HDF5 file stems into train/val/test sets.

    Splits by file (not by chunk) to prevent data leakage.

    Returns
    -------
    Dict with keys "train", "val", "test", each containing a list of file stems.
    """
    with h5py.File(str(hdf5_path), "r") as f:
        all_stems = sorted(f.keys())

    rng = np.random.RandomState(seed)
    rng.shuffle(all_stems)

    n = len(all_stems)
    n_test = max(1, int(n * test_fraction))
    n_val = max(1, int(n * val_fraction))

    test_stems = all_stems[:n_test]
    val_stems = all_stems[n_test:n_test + n_val]
    train_stems = all_stems[n_test + n_val:]

    return {"train": train_stems, "val": val_stems, "test": test_stems}


def chunk_collate_fn(batch: List[Dict]) -> Dict[str, object]:
    """Collate function for DataLoader: stack spectrograms into a batch tensor.

    Returns
    -------
    Dict with "spectrogram" (B, seq_len, n_freq), "file_name" (list), "start_col" (list).
    """
    return {
        "spectrogram": torch.stack([item["spectrogram"] for item in batch]),
        "file_name": [item["file_name"] for item in batch],
        "start_col": [item["start_col"] for item in batch],
    }
