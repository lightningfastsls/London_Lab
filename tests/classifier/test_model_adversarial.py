"""Adversarial tests for model.py — added by test-hardener (Module 18.3).

Targets gaps NOT covered by the 8 original tests:
  A. Degenerate class counts:
     - num_classes=1 (single-class) -> forward shape (B, 1)
     - num_classes=1000 (large) -> forward shape (B, 1000), no OOM on batch=1
  B. Single-batch forward (batch_size=1) — common production edge case
  C. Backward through zeroed loss -> gradients exist but are zero (not NaN)
  D. Independent init: two pretrained=False models have different random
     weights; same torch.manual_seed -> identical weights
  E. Forward produces finite output even when input is all-zeros or all-ones

Total added: 8 tests
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from usv_spectrogram.classifier.model import build_resnet18_classifier  # noqa: E402


# ===========================================================================
# Section A — Degenerate class counts
# ===========================================================================

def test_num_classes_1_forward_shape():
    """num_classes=1 (single-class degenerate case) must produce output shape (B, 1).

    Callers downstream select the argmax across the class dimension. With a
    single class, the model must still produce a valid (B, 1) logit tensor
    rather than collapsing the dimension or raising.
    """
    model = build_resnet18_classifier(num_classes=1, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 227, 227)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 1), (
        f"num_classes=1 forward: expected (2, 1), got {tuple(logits.shape)}"
    )
    assert torch.isfinite(logits).all(), "num_classes=1 produced non-finite logits"


def test_num_classes_1000_forward_shape_no_oom():
    """num_classes=1000 (ImageNet-sized head) must produce shape (1, 1000) on batch=1.

    Tests that the factory correctly wires the classification head for large
    class counts without allocating excessive memory.
    """
    model = build_resnet18_classifier(num_classes=1000, pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 227, 227)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 1000), (
        f"num_classes=1000 forward: expected (1, 1000), got {tuple(logits.shape)}"
    )
    assert torch.isfinite(logits).all(), "num_classes=1000 produced non-finite logits"


# ===========================================================================
# Section B — Single-batch forward
# ===========================================================================

def test_single_sample_forward_shape():
    """Forward on batch_size=1 must produce shape (1, 12) with finite logits.

    Batch-size-1 is used during inference. BatchNorm behaves differently in
    train vs eval mode at batch_size=1 — this test ensures eval mode works.
    """
    model = build_resnet18_classifier(num_classes=12, pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 227, 227)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 12), (
        f"batch_size=1 forward: expected (1, 12), got {tuple(logits.shape)}"
    )
    assert torch.isfinite(logits).all(), (
        "batch_size=1 forward produced non-finite logits"
    )


# ===========================================================================
# Section C — Backward through zeroed loss
# ===========================================================================

def test_backward_through_zeroed_loss_gradients_not_nan():
    """Backprop through a manually-zeroed (multiplied by 0) loss must produce
    zero gradients, not NaN gradients.

    A zeroed loss is mathematically valid: gradients are zero but well-defined.
    NaN gradients here would indicate a broken log(0) or 0/0 in the forward
    pass rather than a true zero-loss scenario.
    """
    model = build_resnet18_classifier(num_classes=12, pretrained=False)
    model.train()
    x = torch.randn(2, 3, 227, 227)
    logits = model(x)
    # Multiply loss by 0 explicitly — gradients should be 0, not NaN.
    loss = logits.sum() * 0.0
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), (
                f"Parameter '{name}' has NaN gradient after backward through "
                "a zero loss. Expected finite (zero) gradients."
            )
            assert (param.grad == 0.0).all(), (
                f"Parameter '{name}' has non-zero gradient after backward "
                "through a zero loss (loss * 0). Expected all-zero gradients."
            )


# ===========================================================================
# Section D — Independent initialisation / seed reproducibility
# ===========================================================================

def test_two_unpretrained_models_have_different_weights():
    """Two calls to build_resnet18_classifier(pretrained=False) without the same
    seed must produce differently initialised weights.

    Shared state (e.g. from a poorly-placed global singleton or class variable)
    would make two independently created models identical, breaking ensemble or
    multiple-run experiments.
    """
    model_a = build_resnet18_classifier(num_classes=12, pretrained=False)
    model_b = build_resnet18_classifier(num_classes=12, pretrained=False)

    # Compare the first convolutional layer weights.
    w_a = next(model_a.parameters()).detach()
    w_b = next(model_b.parameters()).detach()
    # Weights must be different (not identical due to shared state).
    assert not torch.equal(w_a, w_b), (
        "Two independently created pretrained=False models have identical "
        "initial weights. This suggests shared state between instances."
    )


def test_same_seed_gives_identical_weights():
    """Two builds with the same torch.manual_seed must produce identical weights.

    This is the positive-control companion to the test above: determinism
    under a fixed seed is a required property for reproducible experiments.
    """
    torch.manual_seed(2026)
    model_a = build_resnet18_classifier(num_classes=12, pretrained=False)
    torch.manual_seed(2026)
    model_b = build_resnet18_classifier(num_classes=12, pretrained=False)

    w_a = next(model_a.parameters()).detach()
    w_b = next(model_b.parameters()).detach()
    assert torch.equal(w_a, w_b), (
        "Same torch.manual_seed did not produce identical initial weights. "
        "Reproducibility broken — check for internal Random calls outside "
        "the seeded scope."
    )


# ===========================================================================
# Section E — Special-value inputs
# ===========================================================================

def test_forward_all_zeros_input_finite():
    """Forward on an all-zeros input must produce finite logits.

    All-zeros is a natural boundary condition (black spectrogram patch).
    BatchNorm with zero input can produce NaN if variance is zero in train
    mode — eval mode should handle this correctly.
    """
    model = build_resnet18_classifier(num_classes=12, pretrained=False)
    model.eval()
    x = torch.zeros(2, 3, 227, 227)
    with torch.no_grad():
        logits = model(x)
    assert torch.isfinite(logits).all(), (
        f"Forward on all-zeros input produced non-finite logits: "
        f"NaN={torch.isnan(logits).sum()}, Inf={torch.isinf(logits).sum()}"
    )


def test_forward_all_ones_input_finite():
    """Forward on an all-ones input (saturated spectrogram) must produce finite logits.

    All-ones tests saturation clipping through BatchNorm. If any layer
    divides by variance=0 or produces overflow, this catches it.
    """
    model = build_resnet18_classifier(num_classes=12, pretrained=False)
    model.eval()
    x = torch.ones(2, 3, 227, 227)
    with torch.no_grad():
        logits = model(x)
    assert torch.isfinite(logits).all(), (
        f"Forward on all-ones input produced non-finite logits: "
        f"NaN={torch.isnan(logits).sum()}, Inf={torch.isinf(logits).sum()}"
    )
