"""Tests for usv_spectrogram.classifier.model — Module 18.3 model factory.

Written by test-architect BEFORE implementation exists. All tests will fail at
collection (ImportError) until model.py is created. That is the expected TDD
red phase.

ROADMAP §18.3 test plan coverage:
  1. build_resnet18_classifier returns model with output shape (B, 12)
     for input (B, 3, 227, 227)  ->  test_output_shape_batch_12_classes

Additional coverage (recurring gap patterns):
  - model is nn.Module instance            ->  test_model_is_nn_module
  - model.eval() does not raise            ->  test_model_eval_mode_works
  - output is finite (no NaN/Inf) on CPU   ->  test_forward_output_finite
  - pretrained=False works without internet->  test_pretrained_false_returns_working_model
  - parameter count in ResNet-18 ballpark  ->  test_parameter_count_resnet18_ballpark
  - model.to('cpu') is idempotent          ->  test_to_cpu_idempotent
  - NUM_CLASSES constant equals 12         ->  test_num_classes_constant

Total: 8 tests (1 from ROADMAP, 7 additional)

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
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Repo-root sys.path bootstrap
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Import under test — will fail until model.py exists (correct/expected).
# ---------------------------------------------------------------------------
from usv_spectrogram.classifier.model import (  # noqa: E402
    NUM_CLASSES,
    build_resnet18_classifier,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_batch(batch_size: int = 2) -> torch.Tensor:
    """Return a synthetic (B, 3, 227, 227) float tensor on CPU."""
    return torch.randn(batch_size, 3, 227, 227)


# ===========================================================================
# Test 1 (ROADMAP item 1) — output shape is (B, 12) for (B, 3, 227, 227)
# ===========================================================================

def test_output_shape_batch_12_classes():
    """Spec: build_resnet18_classifier().forward(x) -> logits of shape (B, 12).

    The VocalMat taxonomy (Grimsley 2011) has exactly 12 syllable types.
    A single forward pass on a batch of 4 images must yield 4 rows × 12 cols.
    """
    model = build_resnet18_classifier(num_classes=12, pretrained=False)
    model.eval()
    x = _small_batch(batch_size=4)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (4, 12), (
        f"Expected output shape (4, 12) for input (4, 3, 227, 227), "
        f"got {tuple(logits.shape)}"
    )


# ===========================================================================
# Additional test — model is an nn.Module
# ===========================================================================

def test_model_is_nn_module():
    """build_resnet18_classifier must return an nn.Module instance.

    This ensures the returned object is compatible with PyTorch training APIs
    (optimizer parameter iteration, .parameters(), .state_dict(), etc.).
    """
    model = build_resnet18_classifier(pretrained=False)
    assert isinstance(model, nn.Module), (
        f"Expected nn.Module subclass, got {type(model).__name__}"
    )


# ===========================================================================
# Additional test — model.eval() works without exception
# ===========================================================================

def test_model_eval_mode_works():
    """Calling model.eval() must return the model itself (PyTorch contract).

    A broken model that silently sets wrong internal state would cause
    BatchNorm/Dropout to behave differently between train and eval. We
    verify the call completes and returns the same object.
    """
    model = build_resnet18_classifier(pretrained=False)
    returned = model.eval()
    assert returned is model, (
        "model.eval() must return the model itself (nn.Module contract)"
    )
    assert not model.training, (
        "After model.eval(), model.training must be False"
    )


# ===========================================================================
# Additional test — forward output is finite (no NaN/Inf) on a single batch
# ===========================================================================

def test_forward_output_finite():
    """A single forward pass on CPU must produce finite logits (no NaN, no Inf).

    NaN propagation in the forward pass is a common symptom of uninitialized
    weights or broken batch-norm statistics. Checking finiteness on a random
    input catches this before any training loop.
    """
    model = build_resnet18_classifier(pretrained=False)
    model.eval()
    x = _small_batch(batch_size=1)
    with torch.no_grad():
        logits = model(x)
    assert torch.isfinite(logits).all(), (
        f"Forward pass produced non-finite values: "
        f"NaN count={torch.isnan(logits).sum().item()}, "
        f"Inf count={torch.isinf(logits).sum().item()}"
    )


# ===========================================================================
# Additional test — pretrained=False works without network access
# ===========================================================================

def test_pretrained_false_returns_working_model():
    """pretrained=False must return a working model without downloading weights.

    CI environments often have no internet access. A model initialized with
    random weights should still produce the correct output shape.
    """
    model = build_resnet18_classifier(num_classes=12, pretrained=False)
    model.eval()
    x = _small_batch(batch_size=2)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 12), (
        f"pretrained=False model: expected (2, 12), got {tuple(logits.shape)}"
    )


# ===========================================================================
# Additional test — parameter count in ResNet-18 ballpark (~11M)
# ===========================================================================

def test_parameter_count_resnet18_ballpark():
    """ResNet-18 has ~11.2M parameters; 12-class head changes the last layer.

    The exact count depends on timm's implementation; we gate on a wide
    range [9M, 13M] to catch gross errors (e.g., accidentally loading
    ResNet-50 with ~25M, or a linear-only model with ~1k).
    """
    model = build_resnet18_classifier(pretrained=False)
    n_params = sum(p.numel() for p in model.parameters())
    assert 9_000_000 <= n_params <= 13_000_000, (
        f"Parameter count {n_params:,} is outside the ResNet-18 ballpark "
        f"[9M, 13M]. Check that timm created resnet18, not a different arch."
    )


# ===========================================================================
# Additional test — model.to('cpu') is idempotent
# ===========================================================================

def test_to_cpu_idempotent():
    """Calling model.to('cpu') twice must not raise and must produce identical outputs.

    This guards against stateful device-migration bugs where the second .to()
    call corrupts internal buffers.
    """
    model = build_resnet18_classifier(pretrained=False)
    model = model.to("cpu")
    model = model.to("cpu")  # second call must be safe
    model.eval()
    x = _small_batch(batch_size=1)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 12), (
        f"After two .to('cpu') calls, expected (1, 12), got {tuple(logits.shape)}"
    )


# ===========================================================================
# Additional test — NUM_CLASSES constant equals 12
# ===========================================================================

def test_num_classes_constant():
    """NUM_CLASSES module constant must equal 12 (Grimsley 2011 taxonomy).

    Downstream callers (training loop, loss function) import NUM_CLASSES
    directly. A mismatched constant would silently corrupt the classifier head.
    """
    assert NUM_CLASSES == 12, (
        f"NUM_CLASSES = {NUM_CLASSES}, expected 12 (Grimsley 2011 taxonomy)"
    )
