"""Tests for the autoregressive spectrogram transformer.

11 test cases covering:
1. Forward pass shape correctness
2. Causal mask verification (future changes don't affect past)
3. Parameter count within target range
4. Single-batch overfit (proves model can learn)
5. Hidden state extraction shapes
6. Padding mask combined with causal mask (zero gradients for padding)
7. Checkpoint save/resume (optimizer + scheduler state preserved)
8. Gradient clipping (max norm respected)
9-11. TransformerConfig validation (d_model divisibility, negative n_freq, dropout range)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Bootstrap
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usv_language.models.transformer import (
    SpectrogramTransformer,
    TransformerConfig,
)
from usv_language.training.train_transformer import (
    CosineWarmupScheduler,
    build_dataloaders,
    masked_mse_loss,
    save_checkpoint,
    load_checkpoint,
)
from usv_language.data.dataset import AugmentationConfig, TransformerDataConfig
from usv_language.data.dataset import chunk_spectrogram, split_recordings
from usv_language.training.extract_hidden_states import validate_extraction_layers


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> TransformerConfig:
    """Default transformer config (full-size, ~25M params)."""
    return TransformerConfig()


@pytest.fixture
def small_config() -> TransformerConfig:
    """Small config for fast testing (fewer layers/dims)."""
    return TransformerConfig(
        n_freq=170,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ffn=128,
        max_seq_len=32,
        dropout=0.0,  # Deterministic for testing
    )


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Test 1: Forward pass shape
# ---------------------------------------------------------------------------


def test_forward_pass_shape(default_config: TransformerConfig, device: torch.device) -> None:
    """Dummy (4, 128, 170) -> output (4, 128, 170), hidden_states=None."""
    model = SpectrogramTransformer(default_config).to(device)
    model.eval()

    batch_size, seq_len, n_freq = 4, 128, 170
    x = torch.randn(batch_size, seq_len, n_freq, device=device)

    with torch.no_grad():
        output, hidden_states = model(x)

    assert output.shape == (batch_size, seq_len, n_freq), (
        f"Expected ({batch_size}, {seq_len}, {n_freq}), got {output.shape}"
    )
    assert hidden_states is None


# ---------------------------------------------------------------------------
# Test 2: Causal mask verification
# ---------------------------------------------------------------------------


def test_causal_mask(small_config: TransformerConfig, device: torch.device) -> None:
    """Modifying future positions should not change past position outputs.

    We run the model on an input, then modify a future position and re-run.
    Past positions' outputs must be identical (within floating point tolerance).
    """
    model = SpectrogramTransformer(small_config).to(device)
    model.eval()

    seq_len = 16
    x = torch.randn(1, seq_len, small_config.n_freq, device=device)

    with torch.no_grad():
        out_original, _ = model(x)

    # Modify position 12 (future relative to positions 0-11)
    x_modified = x.clone()
    x_modified[0, 12, :] = torch.randn(small_config.n_freq)

    with torch.no_grad():
        out_modified, _ = model(x_modified)

    # Positions 0-11 should be unchanged (they can't see position 12)
    for t in range(12):
        diff = (out_original[0, t] - out_modified[0, t]).abs().max().item()
        assert diff < 1e-5, (
            f"Position {t} changed by {diff} after modifying future position 12"
        )

    # Position 12 itself MAY change (it sees itself via self-attention,
    # but the input at position 12 changed, so it SHOULD change)
    diff_at_12 = (out_original[0, 12] - out_modified[0, 12]).abs().max().item()
    assert diff_at_12 > 1e-3, (
        "Position 12 didn't change after its input was modified — model may be broken"
    )


# ---------------------------------------------------------------------------
# Test 3: Parameter count
# ---------------------------------------------------------------------------


def test_parameter_count(default_config: TransformerConfig) -> None:
    """Full-size model should have 20M < params < 35M."""
    model = SpectrogramTransformer(default_config)
    count = model.count_parameters()
    assert 20_000_000 < count < 35_000_000, (
        f"Parameter count {count:,} outside expected range [20M, 35M]"
    )


# ---------------------------------------------------------------------------
# Test 4: Single-batch overfit
# ---------------------------------------------------------------------------


def test_single_batch_overfit(small_config: TransformerConfig, device: torch.device) -> None:
    """Train 50 steps on one batch; loss should decrease significantly."""
    model = SpectrogramTransformer(small_config).to(device)
    model.train()

    batch_size, seq_len = 4, 16
    x = torch.randn(batch_size, seq_len, small_config.n_freq, device=device)
    target = torch.randn(batch_size, seq_len, small_config.n_freq, device=device)
    mask = torch.ones(batch_size, seq_len, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        output, _ = model(x)
        loss = masked_mse_loss(output, target, mask)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Loss should decrease substantially (at least 50% reduction)
    assert losses[-1] < losses[0] * 0.5, (
        f"Loss didn't decrease enough: {losses[0]:.4f} -> {losses[-1]:.4f} "
        f"(expected at least 50% reduction)"
    )


# ---------------------------------------------------------------------------
# Test 5: Hidden state extraction shapes
# ---------------------------------------------------------------------------


def test_hidden_state_shapes(small_config: TransformerConfig, device: torch.device) -> None:
    """return_hidden_states=True gives n_layers tensors of (B, seq, d_model)."""
    model = SpectrogramTransformer(small_config).to(device)
    model.eval()

    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, small_config.n_freq, device=device)

    with torch.no_grad():
        _, hidden_states = model(x, return_hidden_states=True)

    assert hidden_states is not None
    assert len(hidden_states) == small_config.n_layers, (
        f"Expected {small_config.n_layers} hidden states, got {len(hidden_states)}"
    )
    for i, h in enumerate(hidden_states):
        expected = (batch_size, seq_len, small_config.d_model)
        assert h.shape == expected, (
            f"Layer {i} hidden state shape {h.shape} != expected {expected}"
        )


# ---------------------------------------------------------------------------
# Test 6: Padding mask combined with causal mask
# ---------------------------------------------------------------------------


def test_padding_mask_zero_gradients(small_config: TransformerConfig, device: torch.device) -> None:
    """Padded positions should produce zero gradients when loss is only on valid frames.

    Create a sequence where positions 0-7 are real and 8-15 are padding.
    Compute loss only on valid frames (mask=1). Then check that gradients
    w.r.t. the padded input positions are zero.
    """
    model = SpectrogramTransformer(small_config).to(device)
    model.train()

    seq_len = 16
    valid_len = 8

    x = torch.randn(1, seq_len, small_config.n_freq, device=device, requires_grad=True)
    target = torch.randn(1, seq_len, small_config.n_freq, device=device)

    # Mask: 1 for real (first 8), 0 for padding (last 8)
    mask = torch.zeros(1, seq_len, device=device)
    mask[0, :valid_len] = 1.0

    # Padding mask for model: True where padding
    padding_mask = ~mask.bool()

    output, _ = model(x, attention_mask=padding_mask)
    loss = masked_mse_loss(output, target, mask)
    loss.backward()

    # Gradients for padded input positions should be zero
    # (causal mask prevents looking forward, padding mask prevents attending to pad)
    grad_padded = x.grad[0, valid_len:, :].abs().max().item()
    assert grad_padded < 1e-6, (
        f"Padded positions have non-zero gradient: {grad_padded}"
    )

    # Gradients for valid positions should be non-zero
    grad_valid = x.grad[0, :valid_len, :].abs().max().item()
    assert grad_valid > 1e-6, (
        f"Valid positions have zero gradient: {grad_valid}"
    )


# ---------------------------------------------------------------------------
# Test 7: Checkpoint save/resume
# ---------------------------------------------------------------------------


def test_checkpoint_save_resume(
    small_config: TransformerConfig, device: torch.device, tmp_path: Path,
) -> None:
    """Save state, reload, verify optimizer/scheduler/model preserved."""
    model = SpectrogramTransformer(small_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=10, max_steps=100)

    # Do a few training steps to mutate state
    x = torch.randn(2, 8, small_config.n_freq, device=device)
    for _ in range(5):
        optimizer.zero_grad()
        output, _ = model(x)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Save
    ckpt_path = tmp_path / "test_checkpoint.pt"
    save_checkpoint(
        ckpt_path, model, optimizer, scheduler,
        epoch=5, config=small_config,
        best_val_loss=0.123, history=[{"epoch": 0}],
    )

    # Create fresh model and load
    model2 = SpectrogramTransformer(small_config).to(device)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    scheduler2 = CosineWarmupScheduler(optimizer2, warmup_steps=10, max_steps=100)

    ckpt = load_checkpoint(ckpt_path, model2, optimizer2, scheduler2)

    # Verify checkpoint metadata
    assert ckpt["epoch"] == 5
    assert abs(ckpt["best_val_loss"] - 0.123) < 1e-6

    # Verify scheduler state
    assert scheduler2.step_count == scheduler.step_count

    # Verify model produces same output
    model.eval()
    model2.eval()
    with torch.no_grad():
        out1, _ = model(x)
        out2, _ = model2(x)
    diff = (out1 - out2).abs().max().item()
    assert diff < 1e-6, f"Model outputs differ by {diff} after checkpoint reload"


# ---------------------------------------------------------------------------
# Test 8: Gradient clipping
# ---------------------------------------------------------------------------


def test_gradient_clipping(small_config: TransformerConfig, device: torch.device) -> None:
    """Verify gradient clipping respects max_norm."""
    model = SpectrogramTransformer(small_config).to(device)
    model.train()

    # Create a large-gradient scenario
    x = torch.randn(2, 8, small_config.n_freq, device=device) * 100
    target = torch.randn(2, 8, small_config.n_freq, device=device) * 100

    output, _ = model(x)
    loss = ((output - target) ** 2).mean()
    loss.backward()

    # clip_grad_norm_ returns the ORIGINAL norm before clipping
    max_norm = 1.0
    pre_clip_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_norm,
    )

    # Pre-clip norm should have been large (inputs scaled by 100)
    assert pre_clip_norm > max_norm, (
        f"Pre-clip norm {pre_clip_norm:.4f} was already <= {max_norm}, "
        f"test scenario didn't produce large gradients"
    )

    # Recompute norm AFTER clipping to verify it was capped
    post_clip_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=float("inf"),
    )
    assert post_clip_norm <= max_norm + 1e-3, (
        f"Post-clip grad norm {post_clip_norm:.4f} exceeds max_norm {max_norm}"
    )

    # Verify individual parameter gradients exist (not zeroed out)
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                    for p in model.parameters())
    assert has_grad, "No parameter has non-zero gradient after clipping"


# ---------------------------------------------------------------------------
# TransformerConfig validation tests (bonus)
# ---------------------------------------------------------------------------


class TestTransformerConfigValidation:
    """Test config validation catches invalid parameters."""

    def test_d_model_not_divisible_by_n_heads(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            TransformerConfig(d_model=100, n_heads=3)

    def test_negative_n_freq(self) -> None:
        with pytest.raises(ValueError, match="n_freq"):
            TransformerConfig(n_freq=-1)

    def test_dropout_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="dropout"):
            TransformerConfig(dropout=1.0)


def test_build_dataloaders_requires_nonempty_validation_split() -> None:
    spectrograms = [np.random.randn(170, 40).astype(np.float32)]
    recording_ids = ["rec_001"]
    data_config = TransformerDataConfig(max_seq_len=32, batch_size=2, seed=42)
    aug_config = AugmentationConfig(enabled=False)

    with pytest.raises(ValueError, match="Validation split is empty"):
        build_dataloaders(spectrograms, recording_ids, data_config, aug_config)


def test_chunk_spectrogram_does_not_duplicate_terminal_partial_chunk() -> None:
    spec = np.arange(700 * 3, dtype=np.float32).reshape(700, 3)

    chunks = chunk_spectrogram(spec, max_seq_len=512, overlap_ratio=0.0)

    assert len(chunks) == 2
    assert chunks[0][1] == 512
    assert chunks[1][1] == 188
    np.testing.assert_array_equal(chunks[1][0][:188], spec[512:])


def test_split_recordings_small_dataset_preserves_validation_when_possible() -> None:
    config = TransformerDataConfig(train_split=0.8, val_split=0.1, test_split=0.1, seed=123)

    train_ids, val_ids, test_ids = split_recordings(["a", "b"], config)

    assert len(train_ids) == 1
    assert len(val_ids) == 1
    assert len(test_ids) == 0


def test_validate_extraction_layers_requires_primary_layer_in_requested_layers() -> None:
    with pytest.raises(ValueError, match="primary_layer=4"):
        validate_extraction_layers([2, 6], primary_layer=4, n_layers=8)


# ---------------------------------------------------------------------------
# CosineWarmupScheduler: state roundtrip + LR boundaries
# ---------------------------------------------------------------------------


def test_cosine_scheduler_state_roundtrip() -> None:
    """state_dict/load_state_dict produces identical LR sequences.

    Save state after 15 steps, load into a fresh scheduler with different
    init params, then verify LR matches the original for 20 more steps.
    """
    model = SpectrogramTransformer(TransformerConfig(
        n_freq=170, d_model=64, n_heads=4, n_layers=2, d_ffn=128, max_seq_len=32,
    ))
    opt1 = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched1 = CosineWarmupScheduler(opt1, warmup_steps=10, max_steps=100, min_lr=1e-6)

    # Step 15 times
    for _ in range(15):
        sched1.step()
    saved_state = sched1.state_dict()

    # Create a fresh scheduler with DIFFERENT initial params
    opt2 = torch.optim.AdamW(model.parameters(), lr=5e-4)  # different LR
    sched2 = CosineWarmupScheduler(opt2, warmup_steps=999, max_steps=999, min_lr=0.1)
    sched2.load_state_dict(saved_state)

    # LRs should now match for the next 20 steps
    for step in range(20):
        sched1.step()
        sched2.step()
        lr1 = sched1.get_lr()
        lr2 = sched2.get_lr()
        assert abs(lr1 - lr2) < 1e-10, (
            f"Step {16 + step}: LR mismatch {lr1:.10f} vs {lr2:.10f}"
        )


def test_cosine_scheduler_lr_boundaries() -> None:
    """LR reaches base_lr at end of warmup and ~min_lr at max_steps."""
    model = torch.nn.Linear(10, 10)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = CosineWarmupScheduler(opt, warmup_steps=10, max_steps=100, min_lr=1e-6)

    # Before any step: LR is the initial optimizer LR
    assert abs(opt.param_groups[0]["lr"] - 1e-3) < 1e-10

    # Step through warmup
    for _ in range(10):
        sched.step()
    lr_warmup_end = sched.get_lr()
    assert abs(lr_warmup_end - 1e-3) < 1e-6, (
        f"LR at end of warmup should be ~1e-3, got {lr_warmup_end}"
    )

    # Step to max_steps
    for _ in range(90):
        sched.step()
    lr_end = sched.get_lr()
    assert abs(lr_end - 1e-6) < 1e-7, (
        f"LR at max_steps should be ~1e-6, got {lr_end}"
    )
