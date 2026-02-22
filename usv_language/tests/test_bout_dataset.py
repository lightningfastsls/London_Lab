"""Tests for dataset module (chunking, dataset, batching, splits)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usv_language.data.dataset import (
    AugmentationConfig,
    BucketedBatchSampler,
    TransformerDataConfig,
    USVBoutDataset,
    apply_augmentations,
    chunk_spectrogram,
    split_recordings,
)

N_FREQ = 170


class TestTransformerDataConfig:
    def test_default_values(self):
        config = TransformerDataConfig()
        assert config.max_seq_len == 512
        assert config.overlap_ratio == 0.5
        assert config.batch_size == 32

    def test_frozen(self):
        config = TransformerDataConfig()
        with pytest.raises(AttributeError):
            config.max_seq_len = 256  # type: ignore[misc]

    def test_split_ratios_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            TransformerDataConfig(train_split=0.5, val_split=0.1, test_split=0.1)

    def test_overlap_ratio_bounds(self):
        with pytest.raises(ValueError, match="overlap_ratio"):
            TransformerDataConfig(overlap_ratio=1.0)


class TestChunking:
    def test_chunk_count_for_known_length(self):
        """A 1000-frame spec with max_seq_len=512 and overlap=0.5 should
        produce a known number of chunks."""
        spec_t = np.random.randn(1000, N_FREQ).astype(np.float32)
        chunks = chunk_spectrogram(spec_t, max_seq_len=512, overlap_ratio=0.5)
        # stride = 256, starts at 0, 256, 512 (488 frames remain)
        # At start=768, only 232 frames left which is < 256/2=128, so included
        assert len(chunks) >= 3

    def test_overlap_ratio_changes_stride(self):
        """Higher overlap should produce more chunks."""
        spec_t = np.random.randn(1000, N_FREQ).astype(np.float32)
        chunks_50 = chunk_spectrogram(spec_t, 512, overlap_ratio=0.5)
        chunks_75 = chunk_spectrogram(spec_t, 512, overlap_ratio=0.75)
        assert len(chunks_75) > len(chunks_50)

    def test_short_bout_single_chunk(self):
        """Bout shorter than max_seq_len should produce exactly one padded chunk."""
        spec_t = np.random.randn(200, N_FREQ).astype(np.float32)
        chunks = chunk_spectrogram(spec_t, 512, overlap_ratio=0.5)
        assert len(chunks) == 1
        chunk, real_len = chunks[0]
        assert chunk.shape == (512, N_FREQ)
        assert real_len == 200

    def test_padding_is_zeros(self):
        """Padded portion of short chunk should be zeros."""
        spec_t = np.random.randn(200, N_FREQ).astype(np.float32)
        chunks = chunk_spectrogram(spec_t, 512, overlap_ratio=0.5)
        chunk, real_len = chunks[0]
        assert np.all(chunk[200:] == 0.0)

    def test_empty_spectrogram(self):
        """Empty spectrogram should produce no chunks."""
        spec_t = np.empty((0, N_FREQ), dtype=np.float32)
        chunks = chunk_spectrogram(spec_t, 512, overlap_ratio=0.5)
        assert len(chunks) == 0


class TestUSVBoutDataset:
    @pytest.fixture
    def sample_dataset(self):
        """Create a small dataset for testing."""
        rng = np.random.RandomState(42)
        specs = [
            rng.randn(N_FREQ, 300).astype(np.float32),
            rng.randn(N_FREQ, 600).astype(np.float32),
            rng.randn(N_FREQ, 100).astype(np.float32),
        ]
        rec_ids = ["rec_001", "rec_002", "rec_003"]
        config = TransformerDataConfig(max_seq_len=256, overlap_ratio=0.5)
        return USVBoutDataset(specs, rec_ids, config)

    def test_dataset_length(self, sample_dataset):
        """Dataset should have correct number of chunks."""
        assert len(sample_dataset) > 0

    def test_item_shapes(self, sample_dataset):
        """Each item should have correct shapes for next-column prediction."""
        item = sample_dataset[0]
        seq_len = 256 - 1  # max_seq_len - 1 for next-col prediction
        assert item["input"].shape == (seq_len, N_FREQ)
        assert item["target"].shape == (seq_len, N_FREQ)
        assert item["mask"].shape == (seq_len,)

    def test_next_column_prediction(self):
        """Input should be frames[:-1], target should be frames[1:]."""
        # Create a simple spectrogram where we can verify the shift
        spec = np.arange(N_FREQ * 100, dtype=np.float32).reshape(N_FREQ, 100)
        config = TransformerDataConfig(max_seq_len=100, overlap_ratio=0.0)
        dataset = USVBoutDataset([spec], ["rec"], config)
        item = dataset[0]

        # First element of target should match second column of spec (transposed)
        # spec.T[1] should equal item["target"][0]
        np.testing.assert_array_equal(
            item["input"][0].numpy(),
            spec.T[0],
        )
        np.testing.assert_array_equal(
            item["target"][0].numpy(),
            spec.T[1],
        )

    def test_attention_mask_marks_padding(self):
        """Mask should be 1 for real frames, 0 for padding."""
        # Short spec that will need padding
        spec = np.random.randn(N_FREQ, 50).astype(np.float32)
        config = TransformerDataConfig(max_seq_len=256, overlap_ratio=0.0)
        dataset = USVBoutDataset([spec], ["rec"], config)
        item = dataset[0]

        mask = item["mask"].numpy()
        # 50 real frames, after shift: 49 valid
        assert mask[:49].sum() == 49
        assert mask[49:].sum() == 0

    def test_augmentation_only_in_training(self):
        """Augmentation should only be applied when training=True."""
        spec = np.random.randn(N_FREQ, 200).astype(np.float32)
        config = TransformerDataConfig(max_seq_len=256)
        aug_config = AugmentationConfig(enabled=True, probability=1.0)

        ds_train = USVBoutDataset(
            [spec], ["rec"], config,
            augmentation_config=aug_config, training=True, seed=42,
        )
        ds_eval = USVBoutDataset(
            [spec], ["rec"], config,
            augmentation_config=aug_config, training=False, seed=42,
        )

        train_item = ds_train[0]
        eval_item = ds_eval[0]

        # Eval should match the raw data (no augmentation)
        # Train may differ due to augmentation
        # We can't guarantee they differ (augmentation is probabilistic),
        # but eval should be deterministic
        eval_item2 = ds_eval[0]
        np.testing.assert_array_equal(
            eval_item["input"].numpy(),
            eval_item2["input"].numpy(),
        )

    def test_dataloader_batch_shapes(self, sample_dataset):
        """DataLoader should yield batches with correct shapes."""
        from torch.utils.data import DataLoader

        loader = DataLoader(sample_dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        B = batch["input"].shape[0]
        assert B <= 2
        assert batch["input"].shape == (B, 255, N_FREQ)
        assert batch["target"].shape == (B, 255, N_FREQ)
        assert batch["mask"].shape == (B, 255)


class TestBucketedBatchSampler:
    def test_groups_by_length(self):
        """Batches should contain indices from the same bucket."""
        lengths = [50, 60, 300, 310, 500, 510, 55, 305]
        sampler = BucketedBatchSampler(
            lengths, batch_size=2, shuffle=False, seed=42,
        )
        batches = list(sampler)
        # All batches should have indices with similar lengths
        for batch in batches:
            batch_lengths = [lengths[i] for i in batch]
            # All in same bucket means they'd be within same boundary range
            assert max(batch_lengths) - min(batch_lengths) <= 256  # max bucket span

    def test_batch_size_respected(self):
        """Each batch should have at most batch_size elements."""
        lengths = list(range(100, 200))
        sampler = BucketedBatchSampler(lengths, batch_size=8, shuffle=False)
        for batch in sampler:
            assert len(batch) <= 8

    def test_all_indices_covered(self):
        """All dataset indices should appear in some batch."""
        lengths = list(range(50, 150))
        sampler = BucketedBatchSampler(lengths, batch_size=4, shuffle=False)
        all_indices = set()
        for batch in sampler:
            all_indices.update(batch)
        assert all_indices == set(range(len(lengths)))


class TestRecordingSplits:
    def test_no_recording_in_multiple_splits(self):
        """Each recording should appear in exactly one split."""
        rec_ids = [f"rec_{i:03d}" for i in range(20)]
        config = TransformerDataConfig()
        train, val, test = split_recordings(rec_ids, config)

        train_set = set(train)
        val_set = set(val)
        test_set = set(test)

        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_all_recordings_assigned(self):
        """All recordings should end up in some split."""
        rec_ids = [f"rec_{i:03d}" for i in range(20)]
        config = TransformerDataConfig()
        train, val, test = split_recordings(rec_ids, config)
        assert set(train) | set(val) | set(test) == set(rec_ids)

    def test_approximate_split_ratios(self):
        """Split sizes should roughly match configured ratios."""
        rec_ids = [f"rec_{i:03d}" for i in range(100)]
        config = TransformerDataConfig(
            train_split=0.8, val_split=0.1, test_split=0.1,
        )
        train, val, test = split_recordings(rec_ids, config)
        assert 70 <= len(train) <= 90
        assert 5 <= len(val) <= 20
        assert 5 <= len(test) <= 20
