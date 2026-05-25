"""Tests for usv_spectrogram.classifier.training — Module 18.3 training loop.

Written by test-architect BEFORE implementation exists. All tests will fail at
collection (ImportError) until training.py is created. That is the expected TDD
red phase.

ROADMAP §18.3 test plan coverage:
  8.  TrainingConfig __post_init__ rejects epochs <= 0, batch_size <= 0, lr <= 0
                                           ->  test_trainingconfig_rejects_nonpositive_epochs
                                               test_trainingconfig_rejects_nonpositive_batch_size
                                               test_trainingconfig_rejects_nonpositive_lr
  9.  End-to-end smoke: 12-class synthetic dataset (10 samples per class) trains
      2 epochs in <90s on CPU without NaN losses, produces a checkpoint
                                           ->  test_smoke_train_2_epochs_cpu
  10. Held-out 845 evaluation function returns dict with keys: usv_noise_acc,
      syllable_entropy_mean               ->  test_held_out_eval_returns_required_keys

Additional coverage (recurring gap patterns):
  - TrainingConfig is frozen (immutable)           ->  test_trainingconfig_is_frozen
  - rejects warmup_epochs > epochs                 ->  test_trainingconfig_rejects_warmup_gt_epochs
  - device='auto' resolves to 'cpu' without CUDA   ->  test_trainingconfig_auto_device_resolves_cpu
  - smoke train produces dict with macro_f1_val    ->  test_smoke_train_returns_required_metric_keys
  - syllable_entropy_mean is in valid entropy range ->  test_held_out_entropy_in_valid_range

Total: 10 tests (5 from ROADMAP, 5 additional)

Fixture note — Grimsley 12-class mapping (snake-case folder names):
  Display name          Snake-case folder
  "Noise"           ->  "noise"
  "Step up"         ->  "step_up"
  "Down-FM"         ->  "down_fm"
  "Short"           ->  "short"
  "Chevron"         ->  "chevron"
  "Up-FM"           ->  "up_fm"
  "Flat"            ->  "flat"
  "Two steps"       ->  "two_steps"
  "Step down"       ->  "step_down"
  "Complex"         ->  "complex"
  "Reverse Chevron" ->  "rev_chevron"
  "Multi-steps"     ->  "mult_steps"

Spec ambiguity resolved:
  - syllable_entropy_mean: the spec says "<=log(6)" for the exit criterion on
    real data but does not define the range for synthetic data. We test that
    it is a float in [0, log(12)] — the valid range for a 12-class uniform
    entropy — because the maximum possible entropy over 12 classes is log(12).
  - TrainingConfig.device='auto': not explicitly specified in the ROADMAP
    signature (which shows device='cuda'). We test that 'auto' is supported
    and resolves to 'cpu' on this host (torch.cuda.is_available() == False).
  - held_out_845_csv: the smoke test uses a stub CSV with the expected schema
    (columns: call_label, usv_verdict) rather than the real 845-row file.
"""

from __future__ import annotations

import dataclasses
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Repo-root sys.path bootstrap
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Import under test — will fail until training.py exists (expected).
# ---------------------------------------------------------------------------
from usv_spectrogram.classifier.training import (  # noqa: E402
    TrainingConfig,
    train_classifier,
)

# ---------------------------------------------------------------------------
# Shared synthetic dataset helpers
# ---------------------------------------------------------------------------
_NUM_CLASSES = 12
_IMG_SHAPE = (3, 227, 227)
_SAMPLES_PER_CLASS = 10


def _make_synthetic_loaders(
    samples_per_class: int = _SAMPLES_PER_CLASS,
    batch_size: int = 20,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create synthetic train/val/test DataLoaders with (3, 227, 227) images.

    Each class has ``samples_per_class`` examples. Labels are integers 0..11.
    Uses TensorDataset so no disk I/O is needed.
    """
    torch.manual_seed(seed)
    n_total = samples_per_class * _NUM_CLASSES
    images = torch.randn(n_total, *_IMG_SHAPE)
    labels = torch.arange(_NUM_CLASSES).repeat(samples_per_class)

    # 60/20/20 split among the synthetic samples.
    n_train = int(0.6 * n_total)
    n_val = int(0.2 * n_total)

    train_ds = TensorDataset(images[:n_train], labels[:n_train])
    val_ds = TensorDataset(images[n_train:n_train + n_val], labels[n_train:n_train + n_val])
    test_ds = TensorDataset(images[n_train + n_val:], labels[n_train + n_val:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, val_loader, test_loader


def _make_stub_held_out_csv(tmp_path: Path, n_rows: int = 30) -> Path:
    """Write a minimal held-out CSV with required schema columns.

    Schema mirrors classified_detections_lab_131204_clean.csv:
    - call_label: one of the Grimsley display names or 'noise'
    - usv_verdict: 'usv' or 'noise' (dual-rater verdict)
    """
    rng = np.random.default_rng(42)
    grimsley_classes = [
        "Noise", "Step up", "Down-FM", "Short", "Chevron", "Up-FM",
        "Flat", "Two steps", "Step down", "Complex", "Reverse Chevron",
        "Multi-steps",
    ]
    rows = {
        "call_label": rng.choice(grimsley_classes, size=n_rows).tolist(),
        "usv_verdict": rng.choice(["usv", "noise"], size=n_rows).tolist(),
    }
    csv_path = tmp_path / "stub_held_out_845.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


# ===========================================================================
# Test 8 (ROADMAP item 8) — TrainingConfig rejects nonpositive epochs / batch /lr
# ===========================================================================

def test_trainingconfig_rejects_nonpositive_epochs():
    """Spec: TrainingConfig __post_init__ raises when epochs <= 0.

    Training for 0 or negative epochs is a programmer error that should be
    caught at config construction time, not silently produce an empty result.
    """
    with pytest.raises((ValueError, AssertionError)):
        TrainingConfig(epochs=0)

    with pytest.raises((ValueError, AssertionError)):
        TrainingConfig(epochs=-1)


def test_trainingconfig_rejects_nonpositive_batch_size():
    """Spec: TrainingConfig __post_init__ raises when batch_size <= 0.

    A batch size of 0 would cause division-by-zero in gradient averaging.
    """
    with pytest.raises((ValueError, AssertionError)):
        TrainingConfig(batch_size=0)

    with pytest.raises((ValueError, AssertionError)):
        TrainingConfig(batch_size=-8)


def test_trainingconfig_rejects_nonpositive_lr():
    """Spec: TrainingConfig __post_init__ raises when learning_rate <= 0.

    A non-positive learning rate would either do nothing (lr=0) or invert
    gradient descent (lr<0), both of which are silent bugs.
    """
    with pytest.raises((ValueError, AssertionError)):
        TrainingConfig(learning_rate=0.0)

    with pytest.raises((ValueError, AssertionError)):
        TrainingConfig(learning_rate=-1e-3)


# ===========================================================================
# Test 9 (ROADMAP item 9) — End-to-end smoke: 2 epochs, <90s CPU, checkpoint
# ===========================================================================

def test_smoke_train_2_epochs_cpu(tmp_path):
    """Spec: 12-class synthetic dataset (10 samples/class) trains 2 epochs in
    <90s on CPU without NaN losses and produces a best.pt checkpoint.

    This is the primary integration test for the entire training loop. We use
    a tiny synthetic dataset (120 total samples) and only 2 epochs so the test
    completes well within 90s even on a slow CI host.
    """
    train_loader, val_loader, test_loader = _make_synthetic_loaders(
        samples_per_class=10,
        batch_size=20,
    )
    held_out_csv = _make_stub_held_out_csv(tmp_path)

    cfg = TrainingConfig(
        epochs=2,
        batch_size=20,
        learning_rate=1e-3,
        warmup_epochs=0,
        early_stop_patience=100,  # disable early stopping for this test
        device="cpu",
    )

    t0 = time.monotonic()
    metrics = train_classifier(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        cfg=cfg,
        output_dir=tmp_path,
        held_out_845_csv=held_out_csv,
    )
    elapsed = time.monotonic() - t0

    assert elapsed < 90.0, (
        f"Smoke test took {elapsed:.1f}s, must finish in <90s on CPU. "
        "Investigate expensive operations (loading pretrained weights every step, etc.)."
    )

    checkpoint = tmp_path / "best.pt"
    assert checkpoint.exists(), (
        f"Expected checkpoint at {checkpoint} after training, but file was not created."
    )

    assert metrics is not None and isinstance(metrics, dict), (
        "train_classifier must return a dict of metrics, got "
        f"{type(metrics).__name__}"
    )

    # Verify no NaN losses — checked via the returned metrics dict.
    for key, val in metrics.items():
        if isinstance(val, float):
            assert not math.isnan(val), (
                f"Metric '{key}' is NaN — training produced a NaN loss at some point."
            )


# ===========================================================================
# Test 10 (ROADMAP item 10) — held-out 845 eval returns required dict keys
# ===========================================================================

def test_held_out_eval_returns_required_keys(tmp_path):
    """Spec: the held-out 845 evaluation returns a dict with at least
    'usv_noise_acc' and 'syllable_entropy_mean'.

    These two keys are explicit ROADMAP exit criteria (line 559-560).
    Any downstream reporter that tries to access them without verifying
    presence will produce a KeyError in production — catch it here.
    """
    train_loader, val_loader, test_loader = _make_synthetic_loaders(
        samples_per_class=10,
        batch_size=20,
    )
    held_out_csv = _make_stub_held_out_csv(tmp_path)

    cfg = TrainingConfig(
        epochs=1,
        batch_size=20,
        learning_rate=1e-3,
        warmup_epochs=0,
        early_stop_patience=100,
        device="cpu",
    )

    metrics = train_classifier(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        cfg=cfg,
        output_dir=tmp_path,
        held_out_845_csv=held_out_csv,
    )

    assert "usv_noise_acc" in metrics, (
        f"metrics dict missing 'usv_noise_acc'. Got keys: {sorted(metrics.keys())}"
    )
    assert "syllable_entropy_mean" in metrics, (
        f"metrics dict missing 'syllable_entropy_mean'. Got keys: {sorted(metrics.keys())}"
    )


# ===========================================================================
# Additional test — TrainingConfig is frozen (immutable)
# ===========================================================================

def test_trainingconfig_is_frozen():
    """TrainingConfig must be a frozen dataclass (immutable after construction).

    Config objects are passed into training workers. Mutability creates subtle
    bugs where one worker modifies shared state mid-run.
    """
    cfg = TrainingConfig(epochs=5, batch_size=32, learning_rate=1e-3, device="cpu")
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
        cfg.epochs = 999  # type: ignore[misc]


# ===========================================================================
# Additional test — rejects warmup_epochs > epochs
# ===========================================================================

def test_trainingconfig_rejects_warmup_gt_epochs():
    """TrainingConfig must raise when warmup_epochs > epochs.

    A warmup phase longer than the total training run means the cosine
    scheduler never activates, producing a constant LR for the entire run —
    an undetectable silent bug without this guard.
    """
    with pytest.raises((ValueError, AssertionError)):
        TrainingConfig(epochs=3, warmup_epochs=5, device="cpu")


# ===========================================================================
# Additional test — device='auto' resolves to 'cpu' when no CUDA
# ===========================================================================

def test_trainingconfig_auto_device_resolves_cpu():
    """device='auto' must resolve to 'cpu' on a host without CUDA.

    This test runs on the CI host where torch.cuda.is_available() == False.
    The training loop should detect this and fall back to CPU rather than
    crashing with a CUDA device error.
    """
    assert not torch.cuda.is_available(), (
        "This test assumes no CUDA — if CUDA is available the assertion is moot"
    )
    cfg = TrainingConfig(epochs=1, batch_size=4, learning_rate=1e-3, device="auto")
    # The config itself may store 'auto' and resolve at runtime — we test the
    # resolved device by running one step of training and checking it doesn't crash.
    train_loader, val_loader, test_loader = _make_synthetic_loaders(
        samples_per_class=2, batch_size=4
    )
    tmp = Path("/tmp/test_auto_device")
    tmp.mkdir(exist_ok=True)
    held_out_csv = _make_stub_held_out_csv(tmp)
    # Should not raise RuntimeError: "Expected all tensors to be on CUDA"
    metrics = train_classifier(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        cfg=cfg,
        output_dir=tmp,
        held_out_845_csv=held_out_csv,
    )
    assert isinstance(metrics, dict), (
        "train_classifier with device='auto' must return a dict of metrics"
    )


# ===========================================================================
# Additional test — smoke train returns all required metric keys
# ===========================================================================

def test_smoke_train_returns_required_metric_keys(tmp_path):
    """train_classifier must return a dict containing all PLAN validation keys.

    PLAN §'Validation criteria' requires: macro_f1_val, macro_f1_test,
    per_class_precision, per_class_recall, confusion_matrix.
    Additionally the held-out keys usv_noise_acc and syllable_entropy_mean
    must be present (ROADMAP spec #10).
    """
    train_loader, val_loader, test_loader = _make_synthetic_loaders(
        samples_per_class=5, batch_size=10
    )
    held_out_csv = _make_stub_held_out_csv(tmp_path)

    cfg = TrainingConfig(
        epochs=1,
        batch_size=10,
        learning_rate=1e-3,
        warmup_epochs=0,
        early_stop_patience=100,
        device="cpu",
    )

    metrics = train_classifier(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        cfg=cfg,
        output_dir=tmp_path,
        held_out_845_csv=held_out_csv,
    )

    required_keys = {
        "macro_f1_val",
        "macro_f1_test",
        "per_class_precision",
        "per_class_recall",
        "confusion_matrix",
        "usv_noise_acc",
        "syllable_entropy_mean",
    }
    missing = required_keys - set(metrics.keys())
    assert not missing, (
        f"train_classifier metrics dict missing required keys: {sorted(missing)}. "
        f"Present keys: {sorted(metrics.keys())}"
    )


# ===========================================================================
# Additional test — syllable_entropy_mean is in valid entropy range
# ===========================================================================

def test_held_out_entropy_in_valid_range(tmp_path):
    """syllable_entropy_mean must be a float in [0, log(12)].

    The maximum possible entropy over 12 classes is log(12) ~= 2.485 nats.
    A value outside this range indicates a computation error (e.g., entropy
    computed in bits instead of nats, or negative probability from softmax bug).

    We do not require it to be small (the synthetic dataset is meaningless
    for biological entropy); we only require it to be a physically valid
    entropy value.
    """
    train_loader, val_loader, test_loader = _make_synthetic_loaders(
        samples_per_class=5, batch_size=10
    )
    held_out_csv = _make_stub_held_out_csv(tmp_path)

    cfg = TrainingConfig(
        epochs=1,
        batch_size=10,
        learning_rate=1e-3,
        warmup_epochs=0,
        early_stop_patience=100,
        device="cpu",
    )

    metrics = train_classifier(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        cfg=cfg,
        output_dir=tmp_path,
        held_out_845_csv=held_out_csv,
    )

    entropy = metrics["syllable_entropy_mean"]
    max_entropy = math.log(_NUM_CLASSES)  # log(12) ~= 2.485 nats

    assert isinstance(entropy, float), (
        f"syllable_entropy_mean must be a float, got {type(entropy).__name__}"
    )
    assert 0.0 <= entropy <= max_entropy, (
        f"syllable_entropy_mean = {entropy:.4f} is outside valid range "
        f"[0, log(12)] = [0, {max_entropy:.4f}]. "
        "Check that entropy is computed in nats over the 12 Grimsley classes."
    )
