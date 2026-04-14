"""Integration tests for the transformer training pipeline.

Covers the critical codepaths that will run on the GPU rig:
- train_one_epoch / validate (the actual epoch loops)
- build_optimizer (parameter group assignment)
- masked_mse_loss (numerical correctness + edge cases)
- load_bout_spectrograms (HDF5 + npy loading)
- coerce_transformer_config (checkpoint compat)
- Checkpoint save/resume with scheduler LR continuity
- Recording-level split integrity (no data leakage)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from usv_language.data.dataset import (
    AugmentationConfig,
    BucketedBatchSampler,
    TransformerDataConfig,
    USVBoutDataset,
    split_recordings,
)
from usv_language.models.transformer import SpectrogramTransformer, TransformerConfig
from usv_language.training.train_transformer import (
    CosineWarmupScheduler,
    build_dataloaders,
    build_optimizer,
    coerce_transformer_config,
    load_bout_spectrograms,
    load_checkpoint,
    masked_mse_loss,
    save_checkpoint,
    train_one_epoch,
    validate,
)

N_FREQ = 170


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_config() -> TransformerConfig:
    """Small config for fast CPU testing."""
    return TransformerConfig(
        n_freq=N_FREQ,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ffn=128,
        max_seq_len=32,
        dropout=0.0,
    )


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def synthetic_training_data() -> tuple[list[np.ndarray], list[str]]:
    """5 spectrograms from 3 recordings — enough for a train/val split."""
    rng = np.random.RandomState(42)
    specs = [
        rng.randn(N_FREQ, 200).astype(np.float32),
        rng.randn(N_FREQ, 300).astype(np.float32),
        rng.randn(N_FREQ, 150).astype(np.float32),
        rng.randn(N_FREQ, 250).astype(np.float32),
        rng.randn(N_FREQ, 180).astype(np.float32),
    ]
    rec_ids = ["rec_001", "rec_001", "rec_002", "rec_002", "rec_003"]
    return specs, rec_ids


@pytest.fixture
def small_dataloader_pair(
    synthetic_training_data: tuple[list[np.ndarray], list[str]],
    small_config: TransformerConfig,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders from synthetic data."""
    specs, rec_ids = synthetic_training_data
    data_config = TransformerDataConfig(
        max_seq_len=small_config.max_seq_len,
        batch_size=4,
        num_workers=0,
        seed=42,
    )
    aug_config = AugmentationConfig(enabled=False)
    return build_dataloaders(specs, rec_ids, data_config, aug_config)


# ---------------------------------------------------------------------------
# Tests 1-3: train_one_epoch / validate
# ---------------------------------------------------------------------------


def test_train_one_epoch_returns_finite_loss(
    small_config: TransformerConfig,
    device: torch.device,
    small_dataloader_pair: tuple[DataLoader, DataLoader],
) -> None:
    """train_one_epoch runs without crash and returns a finite positive loss."""
    train_loader, _ = small_dataloader_pair
    model = SpectrogramTransformer(small_config).to(device)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.01)
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=5, max_steps=100,
    )

    loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, grad_clip=1.0)

    assert isinstance(loss, float)
    assert math.isfinite(loss)
    assert loss > 0


def test_train_one_epoch_loss_decreases(
    small_config: TransformerConfig,
    device: torch.device,
    small_dataloader_pair: tuple[DataLoader, DataLoader],
) -> None:
    """Training loss decreases over 3 epochs on the same data."""
    train_loader, _ = small_dataloader_pair
    model = SpectrogramTransformer(small_config).to(device)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.01)
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=2, max_steps=50,
    )

    losses = []
    for _ in range(3):
        loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, grad_clip=1.0,
        )
        losses.append(loss)

    assert losses[-1] < losses[0], (
        f"Loss didn't decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"
    )


def test_validate_returns_finite_loss_no_grads(
    small_config: TransformerConfig,
    device: torch.device,
    small_dataloader_pair: tuple[DataLoader, DataLoader],
) -> None:
    """validate() returns finite loss and leaves no gradients on parameters."""
    _, val_loader = small_dataloader_pair
    model = SpectrogramTransformer(small_config).to(device)

    val_loss = validate(model, val_loader, device)

    assert isinstance(val_loss, float)
    assert math.isfinite(val_loss)
    assert val_loss > 0

    # No parameter should have accumulated gradients
    for name, p in model.named_parameters():
        assert p.grad is None, f"Parameter {name} has gradients after validate()"


# ---------------------------------------------------------------------------
# Test 4: Gradient flow — every parameter gets updated
# ---------------------------------------------------------------------------


def test_gradient_flow_all_params_updated(
    small_config: TransformerConfig,
    device: torch.device,
    small_dataloader_pair: tuple[DataLoader, DataLoader],
) -> None:
    """After 1 training epoch, every trainable parameter has changed."""
    train_loader, _ = small_dataloader_pair
    model = SpectrogramTransformer(small_config).to(device)

    # Snapshot initial parameters
    initial_params = {
        name: p.clone().detach()
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.01)
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=2, max_steps=50,
    )
    train_one_epoch(model, train_loader, optimizer, scheduler, device, grad_clip=1.0)

    # Every parameter should have changed
    unchanged = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        diff = (p - initial_params[name]).abs().sum().item()
        if diff == 0.0:
            unchanged.append(name)

    assert len(unchanged) == 0, (
        f"Dead parameters (no gradient flow): {unchanged}"
    )


# ---------------------------------------------------------------------------
# Test 5: build_optimizer param groups (catches Bug #1)
# ---------------------------------------------------------------------------


def test_build_optimizer_param_groups(small_config: TransformerConfig) -> None:
    """build_optimizer correctly assigns LayerNorm weights to no-decay group.

    BUG EXPOSURE: Parameters inside nn.Sequential get names like
    'input_proj.2.weight' and 'output_head.0.weight' — these ARE LayerNorm
    weights but their names don't contain 'norm'. The current name-based
    check misses them, incorrectly applying weight decay.
    """
    model = SpectrogramTransformer(small_config)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.01)

    assert len(optimizer.param_groups) == 2
    decay_group = optimizer.param_groups[0]
    no_decay_group = optimizer.param_groups[1]

    assert decay_group["weight_decay"] == 0.01
    assert no_decay_group["weight_decay"] == 0.0

    # Build a name -> group mapping
    decay_ids = {id(p) for p in decay_group["params"]}
    no_decay_ids = {id(p) for p in no_decay_group["params"]}

    # No parameter should be in both or neither
    total_opt = len(decay_ids) + len(no_decay_ids)
    total_model = sum(1 for p in model.parameters() if p.requires_grad)
    assert total_opt == total_model, "Parameter count mismatch"
    assert len(decay_ids & no_decay_ids) == 0, "Parameter in both groups"

    # Check specific parameters that should NOT have decay
    import torch.nn as nn

    misclassified = []
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            for param_name, param in module.named_parameters():
                full_name = f"{name}.{param_name}" if name else param_name
                if id(param) in decay_ids:
                    misclassified.append(full_name)

    assert len(misclassified) == 0, (
        f"LayerNorm parameters incorrectly receive weight decay: {misclassified}"
    )


# ---------------------------------------------------------------------------
# Tests 6-8: masked_mse_loss
# ---------------------------------------------------------------------------


def test_masked_mse_loss_all_padding_backward(device: torch.device) -> None:
    """backward() on masked_mse_loss with all-zero mask must not crash.

    BUG EXPOSURE: Currently returns torch.tensor(0.0) which has
    requires_grad=False. Calling .backward() raises RuntimeError.
    """
    pred = torch.randn(2, 8, N_FREQ, device=device, requires_grad=True)
    target = torch.randn(2, 8, N_FREQ, device=device)
    mask = torch.zeros(2, 8, device=device)

    loss = masked_mse_loss(pred, target, mask)
    assert loss.item() == 0.0

    # This should not crash — the loss needs to be differentiable
    try:
        loss.backward()
    except RuntimeError:
        pytest.fail(
            "masked_mse_loss with all-padding mask returned a non-differentiable "
            "tensor. backward() crashed. The returned tensor needs requires_grad=True "
            "or must be connected to the computation graph."
        )


def test_masked_mse_loss_numerical_correctness(device: torch.device) -> None:
    """Hand-computed MSE matches masked_mse_loss output."""
    # 1 batch, 4 frames, 2 freq bins
    pred = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]], device=device)
    target = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], device=device)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]], device=device)

    loss = masked_mse_loss(pred, target, mask)

    # Only first 2 frames count: (1² + 2² + 3² + 4²) / (2 frames × 2 freq) = 30/4 = 7.5
    expected = (1.0 + 4.0 + 9.0 + 16.0) / 4.0
    assert abs(loss.item() - expected) < 1e-6, (
        f"Expected {expected}, got {loss.item()}"
    )


def test_masked_mse_loss_full_mask_equals_mse(device: torch.device) -> None:
    """When mask is all 1s, masked_mse_loss equals standard MSE."""
    pred = torch.randn(4, 16, N_FREQ, device=device)
    target = torch.randn(4, 16, N_FREQ, device=device)
    mask = torch.ones(4, 16, device=device)

    masked_loss = masked_mse_loss(pred, target, mask)
    standard_mse = ((pred - target) ** 2).mean()

    assert torch.allclose(masked_loss, standard_mse, atol=1e-6), (
        f"Masked loss {masked_loss.item():.6f} != standard MSE {standard_mse.item():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 9: coerce_transformer_config
# ---------------------------------------------------------------------------


def test_coerce_transformer_config() -> None:
    """coerce_transformer_config handles None, dict, config, and invalid types."""
    # None -> defaults
    result = coerce_transformer_config(None)
    assert isinstance(result, TransformerConfig)
    assert result == TransformerConfig()

    # dict -> config
    d = {"n_freq": 100, "d_model": 128, "n_heads": 4, "n_layers": 2,
         "d_ffn": 256, "max_seq_len": 64, "dropout": 0.1}
    result = coerce_transformer_config(d)
    assert isinstance(result, TransformerConfig)
    assert result.n_freq == 100

    # config passthrough
    cfg = TransformerConfig(n_freq=50, d_model=64, n_heads=4)
    result = coerce_transformer_config(cfg)
    assert result is cfg

    # invalid type
    with pytest.raises(TypeError, match="Unsupported"):
        coerce_transformer_config(42)


# ---------------------------------------------------------------------------
# Tests 10-12: load_bout_spectrograms
# ---------------------------------------------------------------------------


def test_load_bout_spectrograms_hdf5(tmp_path: Path) -> None:
    """HDF5 loading path returns correct spectrograms and recording IDs."""
    h5_path = tmp_path / "bout_spectrograms.h5"
    rng = np.random.RandomState(42)

    expected_specs = []
    expected_ids = ["rec_A", "rec_A", "rec_B"]

    with h5py.File(h5_path, "w") as f:
        for i, rec_id in enumerate(expected_ids):
            n_frames = 100 + i * 50
            spec = rng.randn(N_FREQ, n_frames).astype(np.float32)
            expected_specs.append(spec)
            grp = f.create_group(f"bout_{i:04d}")
            grp.create_dataset("spectrogram", data=spec)
            grp.attrs["recording_id"] = rec_id

    specs, rec_ids = load_bout_spectrograms(tmp_path)

    assert len(specs) == 3
    assert rec_ids == expected_ids
    for loaded, expected in zip(specs, expected_specs):
        np.testing.assert_array_almost_equal(loaded, expected)


def test_load_bout_spectrograms_npy(tmp_path: Path) -> None:
    """NPY fallback path loads files and parses recording IDs from filenames."""
    rng = np.random.RandomState(42)
    filenames = ["rec001_bout0.npy", "rec001_bout1.npy", "rec002_bout0.npy"]

    for fname in filenames:
        spec = rng.randn(N_FREQ, 100).astype(np.float32)
        np.save(str(tmp_path / fname), spec)

    specs, rec_ids = load_bout_spectrograms(tmp_path)

    assert len(specs) == 3
    # Recording IDs parsed by splitting on "_bout"
    assert rec_ids == ["rec001", "rec001", "rec002"]
    for spec in specs:
        assert spec.dtype == np.float32
        assert spec.shape[0] == N_FREQ


def test_load_bout_spectrograms_empty_raises(tmp_path: Path) -> None:
    """Empty directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="No bout spectrograms found"):
        load_bout_spectrograms(tmp_path)


# ---------------------------------------------------------------------------
# Tests 13-14: Checkpoint resume
# ---------------------------------------------------------------------------


def test_checkpoint_scheduler_lr_continuity(
    small_config: TransformerConfig, device: torch.device, tmp_path: Path,
) -> None:
    """Resumed training produces the same LR schedule as uninterrupted training.

    Train 10 steps, save checkpoint, resume, train 10 more.
    Compare LRs against a continuous 20-step run.
    """
    def make_model_and_scheduler():
        model = SpectrogramTransformer(small_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = CosineWarmupScheduler(
            optimizer, warmup_steps=5, max_steps=30,
        )
        return model, optimizer, scheduler

    # --- Continuous run: 20 steps ---
    model_c, opt_c, sched_c = make_model_and_scheduler()
    continuous_lrs = []
    x = torch.randn(2, 8, small_config.n_freq, device=device)
    for _ in range(20):
        opt_c.zero_grad()
        out, _ = model_c(x)
        out.mean().backward()
        opt_c.step()
        sched_c.step()
        continuous_lrs.append(sched_c.get_lr())

    # --- Interrupted run: 10 steps, save, reload, 10 more ---
    model_i, opt_i, sched_i = make_model_and_scheduler()
    interrupted_lrs = []
    for _ in range(10):
        opt_i.zero_grad()
        out, _ = model_i(x)
        out.mean().backward()
        opt_i.step()
        sched_i.step()
        interrupted_lrs.append(sched_i.get_lr())

    # Save checkpoint
    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt_path, model_i, opt_i, sched_i,
        epoch=0, config=small_config, best_val_loss=999.0,
        history=[], patience_counter=0,
    )

    # Resume
    model_r, opt_r, sched_r = make_model_and_scheduler()
    load_checkpoint(ckpt_path, model_r, opt_r, sched_r)
    for _ in range(10):
        opt_r.zero_grad()
        out, _ = model_r(x)
        out.mean().backward()
        opt_r.step()
        sched_r.step()
        interrupted_lrs.append(sched_r.get_lr())

    # LRs at steps 11-20 should match between continuous and resumed
    for i in range(10, 20):
        assert abs(continuous_lrs[i] - interrupted_lrs[i]) < 1e-8, (
            f"Step {i+1}: continuous LR={continuous_lrs[i]:.8f} != "
            f"resumed LR={interrupted_lrs[i]:.8f}"
        )


def test_checkpoint_preserves_patience(
    small_config: TransformerConfig, device: torch.device, tmp_path: Path,
) -> None:
    """patience_counter survives save/load roundtrip."""
    model = SpectrogramTransformer(small_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=5, max_steps=100)

    ckpt_path = tmp_path / "patience_ckpt.pt"
    save_checkpoint(
        ckpt_path, model, optimizer, scheduler,
        epoch=15, config=small_config, best_val_loss=0.05,
        history=[{"epoch": i} for i in range(16)],
        patience_counter=7,
    )

    ckpt = load_checkpoint(ckpt_path, model)
    assert ckpt["patience_counter"] == 7
    assert ckpt["epoch"] == 15
    assert abs(ckpt["best_val_loss"] - 0.05) < 1e-6


# ---------------------------------------------------------------------------
# Test 15: No recording leakage between train/val
# ---------------------------------------------------------------------------


def test_dataloaders_no_recording_leakage(
    synthetic_training_data: tuple[list[np.ndarray], list[str]],
    small_config: TransformerConfig,
) -> None:
    """No recording ID appears in both train and val dataloaders."""
    specs, rec_ids = synthetic_training_data
    data_config = TransformerDataConfig(
        max_seq_len=small_config.max_seq_len,
        batch_size=4,
        num_workers=0,
        seed=42,
    )
    aug_config = AugmentationConfig(enabled=False)
    train_loader, val_loader = build_dataloaders(
        specs, rec_ids, data_config, aug_config,
    )

    train_recs = set()
    for batch in train_loader:
        train_recs.update(batch["recording_id"])

    val_recs = set()
    for batch in val_loader:
        val_recs.update(batch["recording_id"])

    overlap = train_recs & val_recs
    assert len(overlap) == 0, (
        f"Recording IDs appear in BOTH train and val (data leakage): {overlap}"
    )
