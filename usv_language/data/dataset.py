"""PyTorch dataset and bucketed batch sampler for USV bout spectrograms.

Provides:
- TransformerDataConfig: configuration for chunking and splitting
- USVBoutDataset: PyTorch Dataset that chunks spectrograms into fixed-length
  sequences with next-column prediction targets and attention masks
- BucketedBatchSampler: groups chunks by length into buckets to minimize padding
- Augmentation transforms applied during training
- Recording-based train/val/test splits (ADR-004: no data leakage)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class TransformerDataConfig:
    """Configuration for dataset chunking and data splits."""

    max_seq_len: int = 512
    overlap_ratio: float = 0.5
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 42

    def __post_init__(self) -> None:
        if self.max_seq_len <= 0:
            raise ValueError(
                f"max_seq_len must be positive, got {self.max_seq_len}"
            )
        if not (0.0 <= self.overlap_ratio < 1.0):
            raise ValueError(
                f"overlap_ratio must be in [0, 1), got {self.overlap_ratio}"
            )
        total = self.train_split + self.val_split + self.test_split
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.6f}"
            )


# ---------------------------------------------------------------------------
# Recording-based splits (ADR-004)
# ---------------------------------------------------------------------------


def split_recordings(
    recording_ids: Sequence[str],
    config: TransformerDataConfig,
) -> tuple[list[str], list[str], list[str]]:
    """Split recording IDs into train/val/test sets.

    Splits at the *recording* level to prevent data leakage (ADR-004).
    All bouts from one recording go to the same split.

    Returns (train_ids, val_ids, test_ids).
    """
    ids = sorted(set(recording_ids))
    rng = random.Random(config.seed)
    rng.shuffle(ids)

    n = len(ids)
    if n == 0:
        return [], [], []

    counts = {"train": 1, "val": 0, "test": 0}
    if n >= 2 and config.val_split > 0:
        counts["val"] = 1
    if n >= 3 and config.test_split > 0:
        counts["test"] = 1

    allocated = sum(counts.values())
    if allocated > n:
        overflow = allocated - n
        for split_name in ("test", "val"):
            reducible = min(overflow, counts[split_name])
            counts[split_name] -= reducible
            overflow -= reducible
            if overflow == 0:
                break

    desired = {
        "train": config.train_split * n,
        "val": config.val_split * n,
        "test": config.test_split * n,
    }
    remaining = n - sum(counts.values())
    for _ in range(remaining):
        split_name = max(
            counts.keys(),
            key=lambda name: (desired[name] - counts[name], desired[name]),
        )
        counts[split_name] += 1

    n_train = counts["train"]
    n_val = counts["val"]
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    return train_ids, val_ids, test_ids


# ---------------------------------------------------------------------------
# Augmentation transforms
# ---------------------------------------------------------------------------


@dataclass
class AugmentationConfig:
    """Configuration for training-time augmentation."""

    enabled: bool = True
    probability: float = 0.5
    gaussian_noise_snr_db: float = 17.5
    gain_range_db: tuple[float, float] = (-3.0, 3.0)
    freq_mask_bands: tuple[int, int] = (1, 2)
    freq_mask_width: tuple[int, int] = (20, 30)
    time_mask_count: tuple[int, int] = (1, 2)
    time_mask_ratio: float = 0.1


def apply_augmentations(
    spectrogram: np.ndarray,
    config: AugmentationConfig,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Apply random augmentations to a spectrogram chunk.

    Parameters
    ----------
    spectrogram:
        Input chunk, shape (T, n_freq). Modified in-place and returned.
    config:
        Augmentation parameters.
    rng:
        Random state for reproducibility.

    Returns
    -------
    augmented:
        Augmented spectrogram, same shape as input.
    """
    if not config.enabled:
        return spectrogram

    spec = spectrogram.copy()

    # Gaussian noise
    if rng.random() < config.probability:
        signal_power = np.mean(spec ** 2) + 1e-10
        snr_linear = 10 ** (config.gaussian_noise_snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = rng.randn(*spec.shape).astype(np.float32) * np.sqrt(noise_power)
        spec = spec + noise

    # Gain
    if rng.random() < config.probability:
        gain_db = rng.uniform(config.gain_range_db[0], config.gain_range_db[1])
        spec = spec + gain_db  # In dB domain, gain is additive

    # Frequency masking
    if rng.random() < config.probability:
        n_freq = spec.shape[1]
        n_bands = rng.randint(config.freq_mask_bands[0], config.freq_mask_bands[1] + 1)
        for _ in range(n_bands):
            width = rng.randint(config.freq_mask_width[0], config.freq_mask_width[1] + 1)
            start = rng.randint(0, max(1, n_freq - width))
            end = min(start + width, n_freq)
            spec[:, start:end] = 0.0

    # Time masking
    if rng.random() < config.probability:
        seq_len = spec.shape[0]
        n_masks = rng.randint(config.time_mask_count[0], config.time_mask_count[1] + 1)
        for _ in range(n_masks):
            mask_len = max(1, int(seq_len * config.time_mask_ratio))
            start = rng.randint(0, max(1, seq_len - mask_len))
            end = min(start + mask_len, seq_len)
            spec[start:end, :] = 0.0

    return spec


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


@dataclass
class ChunkInfo:
    """Metadata for a single chunk extracted from a bout spectrogram."""

    bout_index: int
    recording_id: str
    chunk_start_col: int
    real_length: int  # number of real (non-padding) frames


def chunk_spectrogram(
    spectrogram: np.ndarray,
    max_seq_len: int,
    overlap_ratio: float,
) -> list[tuple[np.ndarray, int]]:
    """Chunk a (T, n_freq) spectrogram into fixed-length segments.

    Parameters
    ----------
    spectrogram:
        Input spectrogram transposed to (T, n_freq).
    max_seq_len:
        Maximum chunk length in frames.
    overlap_ratio:
        Overlap between consecutive chunks.

    Returns
    -------
    chunks:
        List of (chunk, real_length) pairs. chunk is shape (max_seq_len, n_freq),
        padded with zeros if needed. real_length is the number of valid frames.
    """
    T, n_freq = spectrogram.shape
    stride = max(1, int(max_seq_len * (1.0 - overlap_ratio)))

    chunks: list[tuple[np.ndarray, int]] = []

    if T == 0:
        return chunks

    if T <= max_seq_len:
        # Single chunk, pad if needed
        chunk = np.zeros((max_seq_len, n_freq), dtype=np.float32)
        chunk[:T] = spectrogram
        chunks.append((chunk, T))
        return chunks

    start = 0
    while start < T:
        end = min(start + max_seq_len, T)
        real_len = end - start
        chunk = np.zeros((max_seq_len, n_freq), dtype=np.float32)
        chunk[:real_len] = spectrogram[start:end]
        chunks.append((chunk, real_len))
        if end == T:
            break

        start += stride
        # Avoid creating a tiny final chunk that's mostly padding
        if start < T and start + max_seq_len < T and T - start < stride // 2:
            # Include remaining in a final overlapping chunk
            chunk = np.zeros((max_seq_len, n_freq), dtype=np.float32)
            chunk_start = max(0, T - max_seq_len)
            actual_data = spectrogram[chunk_start:T]
            chunk[:actual_data.shape[0]] = actual_data
            chunks.append((chunk, actual_data.shape[0]))
            break

    return chunks


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class USVBoutDataset(Dataset):
    """PyTorch dataset for next-column prediction on USV bout spectrograms.

    Each item is a dict with:
        - input: (seq_len, n_freq) — frames[:-1]
        - target: (seq_len, n_freq) — frames[1:]
        - mask: (seq_len,) — 1 for real frames, 0 for padding
        - recording_id: str
    """

    def __init__(
        self,
        spectrograms: list[np.ndarray],
        recording_ids: list[str],
        config: TransformerDataConfig,
        augmentation_config: AugmentationConfig | None = None,
        training: bool = False,
        seed: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        spectrograms:
            List of spectrograms, each shape (n_freq, T_i).
        recording_ids:
            Parallel list of recording IDs (one per spectrogram).
        config:
            Dataset configuration.
        augmentation_config:
            Augmentation config (only used if training=True).
        training:
            Whether this is a training dataset (enables augmentation).
        seed:
            Random seed for augmentation.
        """
        self.config = config
        self.training = training
        self.augmentation_config = augmentation_config or AugmentationConfig(
            enabled=False,
        )
        self.rng = np.random.RandomState(seed)

        # Chunk all spectrograms
        self._chunks: list[tuple[np.ndarray, int]] = []
        self._recording_ids: list[str] = []

        for spec, rec_id in zip(spectrograms, recording_ids):
            # Transpose from (n_freq, T) to (T, n_freq) for chunking
            spec_t = spec.T.astype(np.float32)
            chunks = chunk_spectrogram(
                spec_t, config.max_seq_len, config.overlap_ratio,
            )
            for chunk, real_len in chunks:
                self._chunks.append((chunk, real_len))
                self._recording_ids.append(rec_id)

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int) -> dict:
        chunk, real_len = self._chunks[idx]
        recording_id = self._recording_ids[idx]

        # Apply augmentation (training only)
        if self.training and self.augmentation_config.enabled:
            chunk = apply_augmentations(
                chunk, self.augmentation_config, self.rng,
            )

        # Next-column prediction: input = chunk[:-1], target = chunk[1:]
        input_seq = chunk[:-1]   # (max_seq_len - 1, n_freq)
        target_seq = chunk[1:]   # (max_seq_len - 1, n_freq)

        # Attention mask: 1 for real frames, 0 for padding
        # After shifting, the valid length for input/target is real_len - 1
        valid_len = max(0, real_len - 1)
        mask = np.zeros(input_seq.shape[0], dtype=np.float32)
        mask[:valid_len] = 1.0

        return {
            "input": torch.from_numpy(input_seq),
            "target": torch.from_numpy(target_seq),
            "mask": torch.from_numpy(mask),
            "recording_id": recording_id,
        }

    def get_lengths(self) -> list[int]:
        """Return the real length of each chunk (for bucketed batching)."""
        return [real_len for _, real_len in self._chunks]


# ---------------------------------------------------------------------------
# Bucketed Batch Sampler
# ---------------------------------------------------------------------------


class BucketedBatchSampler(Sampler[list[int]]):
    """Sampler that groups chunks by length into buckets to minimize padding.

    Bucket boundaries: 64, 128, 192, 256, 384, 512
    Within each bucket, samples are shuffled and grouped into batches.
    """

    BUCKET_BOUNDARIES = [64, 128, 192, 256, 384, 512]

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = random.Random(seed)
        self._epoch = 0

    def _assign_bucket(self, length: int) -> int:
        """Assign a length to its bucket index."""
        for i, boundary in enumerate(self.BUCKET_BOUNDARIES):
            if length <= boundary:
                return i
        return len(self.BUCKET_BOUNDARIES) - 1

    def __iter__(self) -> Iterator[list[int]]:
        # Assign indices to buckets
        buckets: dict[int, list[int]] = {}
        for idx, length in enumerate(self.lengths):
            bucket_id = self._assign_bucket(length)
            buckets.setdefault(bucket_id, []).append(idx)

        # Shuffle within each bucket
        if self.shuffle:
            rng = random.Random(self.rng.randint(0, 2**31) + self._epoch)
            for bucket_indices in buckets.values():
                rng.shuffle(bucket_indices)

        # Create batches from each bucket
        all_batches: list[list[int]] = []
        for bucket_indices in buckets.values():
            for i in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[i:i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)

        # Shuffle batch order
        if self.shuffle:
            rng = random.Random(self.rng.randint(0, 2**31) + self._epoch)
            rng.shuffle(all_batches)

        self._epoch += 1
        yield from all_batches

    def __len__(self) -> int:
        # Approximate count
        n_batches = 0
        buckets: dict[int, int] = {}
        for length in self.lengths:
            bucket_id = self._assign_bucket(length)
            buckets[bucket_id] = buckets.get(bucket_id, 0) + 1

        for count in buckets.values():
            if self.drop_last:
                n_batches += count // self.batch_size
            else:
                n_batches += math.ceil(count / self.batch_size)

        return n_batches
