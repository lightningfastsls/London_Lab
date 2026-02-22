"""Tests for SpectrogramChunkDataset and related utilities."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from usv_language.src.data.dataset import (
    SpectrogramChunkDataset,
    create_splits,
    chunk_collate_fn,
)


class TestSpectrogramChunkDataset:
    def test_basic_loading(self, synthetic_hdf5):
        ds = SpectrogramChunkDataset(
            synthetic_hdf5,
            chunk_length=64,
            overlap_fraction=0.0,
            silence_threshold=0.0,
        )
        assert len(ds) > 0

    def test_chunk_shape(self, synthetic_hdf5):
        ds = SpectrogramChunkDataset(
            synthetic_hdf5,
            chunk_length=64,
            overlap_fraction=0.0,
            silence_threshold=0.0,
        )
        item = ds[0]
        assert item["spectrogram"].shape == (64, 170)  # (seq_len, n_freq)
        assert isinstance(item["spectrogram"], torch.Tensor)
        assert isinstance(item["file_name"], str)
        assert isinstance(item["start_col"], int)

    def test_overlap_increases_chunks(self, synthetic_hdf5):
        ds_no_overlap = SpectrogramChunkDataset(
            synthetic_hdf5, chunk_length=64, overlap_fraction=0.0, silence_threshold=0.0,
        )
        ds_with_overlap = SpectrogramChunkDataset(
            synthetic_hdf5, chunk_length=64, overlap_fraction=0.5, silence_threshold=0.0,
        )
        assert len(ds_with_overlap) > len(ds_no_overlap)

    def test_normalization(self, synthetic_hdf5):
        """With correct global norm params, values should be mostly in [0, 1]."""
        from usv_language.src.data.normalization import compute_global_norm_params
        params = compute_global_norm_params(str(synthetic_hdf5))
        ds = SpectrogramChunkDataset(
            synthetic_hdf5,
            chunk_length=64,
            overlap_fraction=0.0,
            silence_threshold=0.0,
            norm_params=(params.min_val, params.max_val),
        )
        item = ds[0]
        # With correct global params, values should be in [0, 1]
        assert item["spectrogram"].min() >= -0.01
        assert item["spectrogram"].max() <= 1.01

    def test_file_stem_filter(self, synthetic_hdf5):
        ds = SpectrogramChunkDataset(
            synthetic_hdf5,
            chunk_length=64,
            overlap_fraction=0.0,
            silence_threshold=0.0,
            file_stems=["file_000"],
        )
        # All chunks should come from file_000
        for i in range(len(ds)):
            assert ds[i]["file_name"] == "file_000"

    def test_invalid_overlap(self, synthetic_hdf5):
        import pytest
        with pytest.raises(ValueError, match="overlap_fraction"):
            SpectrogramChunkDataset(synthetic_hdf5, overlap_fraction=1.0)


class TestCreateSplits:
    def test_splits_cover_all_files(self, synthetic_hdf5):
        splits = create_splits(synthetic_hdf5)
        all_stems = splits["train"] + splits["val"] + splits["test"]
        assert len(all_stems) == 3  # 3 files in synthetic HDF5
        assert len(set(all_stems)) == 3  # No duplicates

    def test_deterministic(self, synthetic_hdf5):
        s1 = create_splits(synthetic_hdf5, seed=42)
        s2 = create_splits(synthetic_hdf5, seed=42)
        assert s1 == s2


class TestCollate:
    def test_collate_fn(self, synthetic_hdf5):
        ds = SpectrogramChunkDataset(
            synthetic_hdf5, chunk_length=64, silence_threshold=0.0,
        )
        loader = DataLoader(ds, batch_size=4, collate_fn=chunk_collate_fn)
        batch = next(iter(loader))
        assert batch["spectrogram"].shape[0] == 4
        assert batch["spectrogram"].shape[1] == 64
        assert batch["spectrogram"].shape[2] == 170
        assert len(batch["file_name"]) == 4
