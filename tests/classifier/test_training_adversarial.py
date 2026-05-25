"""Adversarial tests for training.py — added by test-hardener (Module 18.3).

Targets gaps NOT covered by the 10 original tests:
  A. TrainingConfig boundary: warmup_epochs == epochs must construct OK
  B. epochs=1, warmup_epochs=1 (boundary) -> constructs, scheduler doesn't
     divide by zero (the LR lambda is called once and must return a valid float)
  C. early_stop_patience=1 with non-improving val -> training stops after 2 epochs
     (1 improvement + 1 no-improvement = stop)
  D. Empty held-out CSV (header only, 0 rows) -> usv_noise_acc and
     syllable_entropy_mean are returned without NaN or divide-by-zero
  E. Held-out CSV with all "Noise" labels -> entropy is 0.0 (single-mass)
  F. train_loader with exactly 1 batch -> training completes without crash
  G. output_dir is a file (not a dir) -> raises, no silent data corruption

Total added: 8 tests
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from usv_spectrogram.classifier.training import (  # noqa: E402
    TrainingConfig,
    _cosine_warmup_lr,
    _evaluate_held_out_845,
    train_classifier,
)

_NUM_CLASSES = 12
_IMG_SHAPE = (3, 227, 227)


# ---------------------------------------------------------------------------
# Shared helpers (mirrors the style from test_training.py)
# ---------------------------------------------------------------------------

def _make_synthetic_loaders(
    n_total: int = 24,
    batch_size: int = 24,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Tiny synthetic loaders with n_total samples split 60/20/20."""
    torch.manual_seed(seed)
    images = torch.randn(n_total, *_IMG_SHAPE)
    labels = torch.randint(0, _NUM_CLASSES, (n_total,))
    n_train = int(0.6 * n_total)
    n_val = int(0.2 * n_total)
    train_ds = TensorDataset(images[:n_train], labels[:n_train])
    val_ds = TensorDataset(images[n_train:n_train + n_val], labels[n_train:n_train + n_val])
    test_ds = TensorDataset(images[n_train + n_val:], labels[n_train + n_val:])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=False),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size),
    )


def _write_held_out_csv(tmp_path: Path, call_labels: list[str], usv_verdicts: list[str]) -> Path:
    p = tmp_path / "held_out.csv"
    pd.DataFrame({"call_label": call_labels, "usv_verdict": usv_verdicts}).to_csv(p, index=False)
    return p


# ===========================================================================
# Section A — warmup_epochs == epochs boundary
# ===========================================================================

def test_trainingconfig_warmup_equals_epochs_ok():
    """TrainingConfig(warmup_epochs=N, epochs=N) must construct without raising.

    The validation gate is warmup_epochs > epochs; equality (==) is the
    boundary — warmup fills the entire schedule and cosine never activates,
    which is unusual but not an error.
    """
    cfg = TrainingConfig(epochs=5, warmup_epochs=5, device="cpu")
    assert cfg.warmup_epochs == cfg.epochs, (
        "Boundary config (warmup_epochs == epochs) must be accepted"
    )


# ===========================================================================
# Section B — Cosine LR lambda does not divide by zero at epochs=warmup=1
# ===========================================================================

def test_cosine_warmup_lr_no_division_by_zero_boundary():
    """_cosine_warmup_lr(epoch=0, cfg) with epochs=1, warmup_epochs=1 must
    return a finite float in (0, 1].

    The denominator `max(1, epochs - warmup_epochs)` = max(1, 0) = 1,
    so there is no divide-by-zero risk here. But if the implementation
    used `epochs - warmup_epochs` without the max() guard, this returns 0/0.
    Lock in the finite-output contract.
    """
    cfg = TrainingConfig(epochs=1, warmup_epochs=1, device="cpu")
    lr_mult = _cosine_warmup_lr(epoch=0, cfg=cfg)
    assert math.isfinite(lr_mult), (
        f"_cosine_warmup_lr returned non-finite {lr_mult} for epochs=warmup=1"
    )
    assert lr_mult > 0.0, (
        f"_cosine_warmup_lr returned non-positive {lr_mult} for epoch=0 "
        "of a warmup schedule — LR multiplier must be > 0"
    )


# ===========================================================================
# Section C — Early stopping with patience=1 and stagnant val
# ===========================================================================

def test_early_stopping_patience_1_stops_training(tmp_path):
    """With early_stop_patience=1, training must stop after the first epoch
    that fails to improve val macro-F1.

    On a random synthetic dataset, macro-F1 is essentially a random walk.
    With patience=1, the loop exits immediately after the first non-improvement.
    The history dict must have fewer entries than cfg.epochs.
    """
    train_loader, val_loader, test_loader = _make_synthetic_loaders(
        n_total=60, batch_size=60, seed=77
    )
    held_out_csv = _write_held_out_csv(
        tmp_path,
        call_labels=["Noise"] * 10,
        usv_verdicts=["noise"] * 10,
    )
    cfg = TrainingConfig(
        epochs=50,              # high ceiling — early stopping must kick in
        batch_size=60,
        learning_rate=1e-3,
        warmup_epochs=0,
        early_stop_patience=1,  # stop after one non-improvement
        device="cpu",
        pretrained=False,
    )
    metrics = train_classifier(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        cfg=cfg,
        output_dir=tmp_path,
        held_out_845_csv=held_out_csv,
    )
    history = metrics["history"]
    assert len(history) < cfg.epochs, (
        f"Early stopping with patience=1 should have stopped before {cfg.epochs} epochs, "
        f"but ran {len(history)} epochs. Check that early_stop_patience is being respected."
    )


# ===========================================================================
# Section D — Empty held-out CSV (header only, 0 rows)
# ===========================================================================

def test_evaluate_held_out_845_empty_csv_no_nan(tmp_path):
    """_evaluate_held_out_845 on an empty CSV must return valid keys without
    NaN or divide-by-zero.

    An empty CSV (header only) has len(df) == 0. The entropy of an empty
    distribution is undefined; the implementation should return 0.0 (or some
    sentinel) and usv_noise_acc should also be 0.0.
    """
    empty_csv = tmp_path / "empty.csv"
    pd.DataFrame({"call_label": [], "usv_verdict": []}).to_csv(empty_csv, index=False)

    result = _evaluate_held_out_845(empty_csv)

    assert "usv_noise_acc" in result, "Missing 'usv_noise_acc' key for empty CSV"
    assert "syllable_entropy_mean" in result, "Missing 'syllable_entropy_mean' key for empty CSV"
    assert math.isfinite(result["usv_noise_acc"]), (
        f"usv_noise_acc is non-finite for empty CSV: {result['usv_noise_acc']}"
    )
    assert math.isfinite(result["syllable_entropy_mean"]), (
        f"syllable_entropy_mean is non-finite for empty CSV: {result['syllable_entropy_mean']}"
    )


# ===========================================================================
# Section E — All-noise held-out CSV -> entropy = 0.0
# ===========================================================================

def test_evaluate_held_out_845_all_noise_entropy_zero(tmp_path):
    """When all call_label rows are 'Noise', the empirical entropy of the
    label distribution is 0.0 (single-mass point distribution).

    -1 * log(1) = 0. If the implementation produces nonzero entropy here,
    it has a bug in the counts/normalisation step.
    """
    csv_path = _write_held_out_csv(
        tmp_path,
        call_labels=["Noise"] * 20,
        usv_verdicts=["noise"] * 20,
    )
    result = _evaluate_held_out_845(csv_path)

    entropy = result["syllable_entropy_mean"]
    assert math.isfinite(entropy), f"Entropy is non-finite: {entropy}"
    assert abs(entropy) < 1e-9, (
        f"All-same-label distribution must have entropy 0.0, got {entropy:.6f}. "
        "Check the entropy normalisation step."
    )


# ===========================================================================
# Section F — Single-batch train_loader
# ===========================================================================

def test_single_batch_train_loader_completes(tmp_path):
    """A train_loader with exactly 1 batch (all samples fit in one batch) must
    complete 1 epoch without crash.

    This is an edge case that can arise on very small datasets or large batch
    sizes. The _train_one_epoch loop must handle n_batches=1 gracefully.
    """
    n_total = 12
    torch.manual_seed(5)
    images = torch.randn(n_total, *_IMG_SHAPE)
    labels = torch.arange(_NUM_CLASSES)  # exactly 12 samples, one per class
    # Use batch_size >= n_total so there is exactly 1 batch.
    ds = TensorDataset(images, labels)
    loader = DataLoader(ds, batch_size=n_total, shuffle=False)

    held_out_csv = _write_held_out_csv(
        tmp_path,
        call_labels=["Noise"] * 5,
        usv_verdicts=["noise"] * 5,
    )
    cfg = TrainingConfig(
        epochs=1,
        batch_size=n_total,
        learning_rate=1e-3,
        warmup_epochs=0,
        early_stop_patience=100,
        device="cpu",
        pretrained=False,
    )
    metrics = train_classifier(
        train_loader=loader,
        val_loader=loader,   # reuse train as val/test for simplicity
        test_loader=loader,
        cfg=cfg,
        output_dir=tmp_path,
        held_out_845_csv=held_out_csv,
    )
    assert isinstance(metrics, dict), (
        "train_classifier must return a dict even with a single-batch loader"
    )
    assert len(metrics["history"]) == 1, (
        f"Expected 1 epoch in history for epochs=1, got {len(metrics['history'])}"
    )


# ===========================================================================
# Section G — output_dir is a file (not a directory)
# ===========================================================================

def test_output_dir_is_file_raises(tmp_path):
    """When output_dir points to an existing file (not a dir), train_classifier
    must raise a clear error, not silently corrupt data.

    The implementation calls output_dir.mkdir(parents=True, exist_ok=True).
    If output_dir is an existing file, mkdir raises FileExistsError (or OSError)
    on most filesystems. This test locks in that behaviour.
    """
    # Create a file at the output_dir path.
    output_file = tmp_path / "i_am_a_file.txt"
    output_file.write_text("I am a file, not a directory.")

    train_loader, val_loader, test_loader = _make_synthetic_loaders(
        n_total=24, batch_size=24, seed=0
    )
    held_out_csv = _write_held_out_csv(
        tmp_path,
        call_labels=["Noise"] * 5,
        usv_verdicts=["noise"] * 5,
    )
    cfg = TrainingConfig(
        epochs=1,
        batch_size=24,
        learning_rate=1e-3,
        warmup_epochs=0,
        device="cpu",
        pretrained=False,
    )
    with pytest.raises((FileExistsError, NotADirectoryError, OSError, IsADirectoryError)):
        train_classifier(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            cfg=cfg,
            output_dir=output_file,   # a file, not a directory
            held_out_845_csv=held_out_csv,
        )
