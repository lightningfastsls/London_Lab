"""Adversarial tests for losses.py — added by test-hardener (Module 18.3).

Targets gaps NOT covered by the 7 original tests:
  A. gamma=10 (extreme focal): confident-correct loss is even smaller than gamma=2
  B. All-same-class degenerate batch -> loss is scalar and gradient flows
  C. batch_size=1 single-sample -> finite loss and finite gradient
  D. class_weights with a zero slot for the targeted class -> per-sample loss = 0
     for those samples (and total loss may be 0 if only that class present)
  E. Logits with very large magnitude (+1e6) -> no NaN, no overflow in softmax
  F. Logits with very negative magnitude at target class (-1e6) -> finite loss
     (log(0) guard via log_softmax numerical stability)
  G. Target at class boundaries (0 and NUM_CLASSES-1) -> no off-by-one error

Total added: 8 tests
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from usv_spectrogram.classifier.losses import focal_loss  # noqa: E402

_NUM_CLASSES = 12


def _uniform_weights(n: int = _NUM_CLASSES) -> torch.Tensor:
    return torch.ones(n)


# ===========================================================================
# Section A — Extreme gamma: gamma=10 vs gamma=2 on confident-correct
# ===========================================================================

def test_extreme_gamma_10_smaller_loss_than_gamma_2():
    """With gamma=10, a high-confidence correct prediction gives an even smaller
    loss than with gamma=2.

    The focal term (1-p_t)^gamma decays faster for higher gamma when p_t is
    close to 1. This is the core property of the focusing parameter — it
    aggressively suppresses easy examples for large gamma.
    """
    # Construct a logit vector that's very confident for class 0.
    logits = torch.zeros(1, _NUM_CLASSES)
    logits[0, 0] = 15.0   # p_0 ~ 1.0
    targets = torch.tensor([0])
    weights = _uniform_weights()

    loss_gamma2 = focal_loss(logits, targets, class_weights=weights, gamma=2.0)
    loss_gamma10 = focal_loss(logits, targets, class_weights=weights, gamma=10.0)

    assert loss_gamma10.item() < loss_gamma2.item(), (
        f"gamma=10 loss ({loss_gamma10.item():.6f}) must be smaller than "
        f"gamma=2 loss ({loss_gamma2.item():.6f}) for a high-confidence "
        "correct prediction. Larger gamma suppresses easy examples more."
    )
    assert torch.isfinite(loss_gamma10), (
        f"gamma=10 produced non-finite loss: {loss_gamma10.item()}"
    )


# ===========================================================================
# Section B — All-same-class degenerate batch
# ===========================================================================

def test_all_same_class_batch_scalar_and_gradient_flows():
    """A batch where every sample has the same class label must still produce
    a scalar finite loss with flowing gradients.

    This is the degenerate case that can arise in imbalanced mini-batches
    (e.g., all samples happen to be class 0 in a small batch). If the
    implementation has a bug with homogeneous batches it will show here.
    """
    torch.manual_seed(7)
    logits = torch.randn(8, _NUM_CLASSES, requires_grad=True)
    targets = torch.zeros(8, dtype=torch.long)   # all class 0
    weights = _uniform_weights()

    loss = focal_loss(logits, targets, class_weights=weights, gamma=2.0)

    assert loss.ndim == 0, (
        f"focal_loss must return a scalar, got shape {loss.shape}"
    )
    assert torch.isfinite(loss), (
        f"focal_loss on all-same-class batch returned non-finite: {loss.item()}"
    )

    loss.backward()
    assert logits.grad is not None, "No gradient for logits after backward"
    assert torch.isfinite(logits.grad).all(), (
        "Non-finite gradient on all-same-class batch"
    )


# ===========================================================================
# Section C — Single-sample batch (batch_size=1)
# ===========================================================================

def test_single_sample_batch_finite_loss_and_gradient():
    """batch_size=1 must produce finite loss and finite gradient.

    Single-sample batches occur during online learning and the last mini-batch
    of odd-sized datasets. The plain-mean reduction (sum/B with B=1) must
    work without divide-by-zero or special-case branching.
    """
    torch.manual_seed(13)
    logits = torch.randn(1, _NUM_CLASSES, requires_grad=True)
    targets = torch.tensor([5])
    weights = _uniform_weights()

    loss = focal_loss(logits, targets, class_weights=weights, gamma=2.0)

    assert loss.ndim == 0, (
        f"focal_loss with batch_size=1 must return scalar, got {loss.shape}"
    )
    assert torch.isfinite(loss), (
        f"focal_loss batch_size=1 returned non-finite: {loss.item()}"
    )
    assert loss.item() >= 0.0, (
        f"focal_loss must be non-negative, got {loss.item()}"
    )

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all(), (
        "Non-finite gradient with batch_size=1"
    )


# ===========================================================================
# Section D — Zero weight for the targeted class
# ===========================================================================

def test_zero_weight_for_targeted_class_gives_zero_loss():
    """class_weights[target_class] = 0 must produce zero loss for samples of
    that class, and zero gradients (the weight cancels the per-sample term).

    This tests the per-class scaling directly: weight=0 means that class
    contributes nothing to the total loss. If the implementation accidentally
    uses the weight from a different index, the loss will be nonzero.
    """
    logits = torch.randn(4, _NUM_CLASSES)
    targets = torch.tensor([3, 3, 3, 3])   # all class 3
    weights = _uniform_weights()
    weights[3] = 0.0   # zero weight for the targeted class

    loss = focal_loss(logits, targets, class_weights=weights, gamma=2.0)

    assert loss.ndim == 0, f"focal_loss must return scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), f"Loss is non-finite: {loss.item()}"
    assert loss.item() == 0.0, (
        f"With class_weights[target_class]=0 and all targets = that class, "
        f"focal_loss must be 0.0, got {loss.item():.8f}"
    )


# ===========================================================================
# Section E — Very large logit magnitude (potential overflow)
# ===========================================================================

def test_very_large_logit_magnitude_no_nan():
    """Logits with magnitude +1e6 must not produce NaN or Inf.

    F.log_softmax uses the log-sum-exp trick internally which is numerically
    stable for large positive values, but a naive implementation (log(softmax))
    would overflow to NaN. This test guards against that regression.
    """
    logits = torch.zeros(2, _NUM_CLASSES)
    logits[0, 0] = 1e6     # extremely large score for class 0
    logits[1, 5] = 1e6     # extremely large score for class 5
    targets = torch.tensor([0, 5])
    weights = _uniform_weights()

    loss = focal_loss(logits, targets, class_weights=weights, gamma=2.0)

    assert torch.isfinite(loss), (
        f"focal_loss with logit magnitude 1e6 returned non-finite: {loss.item()}"
    )
    assert loss.item() >= 0.0, (
        f"focal_loss must be non-negative, got {loss.item()}"
    )


# ===========================================================================
# Section F — Very negative logit at target class (potential log(0))
# ===========================================================================

def test_very_negative_target_logit_no_nan():
    """When the target class logit is -1e6, log(p_t) approaches log(0).
    The loss must remain finite (large but not NaN/Inf).

    F.log_softmax handles this numerically by clamping internally, but a
    naive log(softmax) approach would produce log(0) = -inf -> NaN via
    the focal-term multiplication (0 * -inf = NaN).
    """
    logits = torch.zeros(2, _NUM_CLASSES)
    logits[0, 1] = -1e6    # target class gets a devastating score
    logits[1, 3] = -1e6    # same for second sample
    targets = torch.tensor([1, 3])
    weights = _uniform_weights()

    loss = focal_loss(logits, targets, class_weights=weights, gamma=2.0)

    assert torch.isfinite(loss), (
        f"focal_loss with very negative target logit returned non-finite: "
        f"{loss.item()}. This usually indicates log(0) -> NaN."
    )
    assert loss.item() >= 0.0, (
        f"focal_loss must be non-negative, got {loss.item()}"
    )


# ===========================================================================
# Section G — Target at class boundaries (off-by-one guard)
# ===========================================================================

def test_target_at_class_boundary_0():
    """Target class 0 (lowest valid class index) must produce finite loss.

    Off-by-one errors in index gathering (e.g., using 1-indexed classes)
    would produce an index-out-of-bounds or silently gather the wrong logit.
    """
    logits = torch.randn(3, _NUM_CLASSES)
    targets = torch.tensor([0, 0, 0])   # all boundary class 0
    weights = _uniform_weights()

    loss = focal_loss(logits, targets, class_weights=weights, gamma=2.0)

    assert loss.ndim == 0
    assert torch.isfinite(loss), (
        f"focal_loss with target=0 (boundary) returned non-finite: {loss.item()}"
    )


def test_target_at_class_boundary_max():
    """Target class NUM_CLASSES-1 (highest valid class index) must produce finite loss.

    The last class (index 11, Multi-steps) is the rarest in the dataset.
    Off-by-one in index gathering with 0-indexed tensors would access index 12,
    which is out of range.
    """
    logits = torch.randn(3, _NUM_CLASSES)
    targets = torch.tensor([11, 11, 11])   # all boundary class 11
    weights = _uniform_weights()

    loss = focal_loss(logits, targets, class_weights=weights, gamma=2.0)

    assert loss.ndim == 0
    assert torch.isfinite(loss), (
        f"focal_loss with target={_NUM_CLASSES - 1} (boundary) returned "
        f"non-finite: {loss.item()}"
    )
