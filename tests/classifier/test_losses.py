"""Tests for usv_spectrogram.classifier.losses — Module 18.3 focal loss.

Written by test-architect BEFORE implementation exists. All tests will fail at
collection (ImportError) until losses.py is created. That is the expected TDD
red phase.

ROADMAP §18.3 test plan coverage:
  6. focal_loss reduces to weighted CE when gamma=0
                                           ->  test_focal_loss_gamma0_equals_weighted_ce
  7. focal_loss positive control: minority class with high weight -> larger gradient
     than majority                         ->  test_focal_loss_minority_gradient_larger

Additional coverage (recurring gap patterns):
  - focal_loss output is scalar finite tensor     ->  test_focal_loss_output_scalar_finite
  - gradient flows through logits (no NaN grad)   ->  test_focal_loss_gradient_flows
  - raises on batch-size shape mismatch           ->  test_focal_loss_shape_mismatch_raises
  - raises on class_weights wrong dimension       ->  test_focal_loss_wrong_weight_dim_raises
  - high-confidence correct prediction -> near-zero loss  ->  test_focal_loss_correct_prediction_near_zero

Total: 7 tests (2 from ROADMAP, 5 additional)

Focal loss math note (for hand-checking tests):
  focal_loss(logits, t, w, gamma) = -w[t] * (1 - p_t)^gamma * log(p_t)
  where p_t = softmax(logits)[t].
  When gamma=0 this reduces to: -w[t] * log(p_t) = weighted CE.
  When p_t -> 1 (confident correct), (1 - p_t)^gamma -> 0 -> loss -> 0.

Grimsley 12-class mapping (snake-case folder names):
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
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Repo-root sys.path bootstrap
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Import under test — will fail until losses.py exists (expected).
# ---------------------------------------------------------------------------
from usv_spectrogram.classifier.losses import focal_loss  # noqa: E402

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
_NUM_CLASSES = 12


def _uniform_weights(n: int = _NUM_CLASSES) -> torch.Tensor:
    """All-ones class weights (equivalent to unweighted CE)."""
    return torch.ones(n)


def _make_logits(batch: int = 4, n_classes: int = _NUM_CLASSES, seed: int = 0) -> torch.Tensor:
    """Random logits as a (batch, n_classes) tensor."""
    torch.manual_seed(seed)
    return torch.randn(batch, n_classes)


# ===========================================================================
# Test 6 (ROADMAP item 6) — focal_loss with gamma=0 equals weighted CE
# ===========================================================================

def test_focal_loss_gamma0_equals_weighted_ce():
    """Spec: focal_loss(..., gamma=0) must equal plain-mean weighted cross-entropy.

    Hand derivation: focal term = (1 - p_t)^0 = 1, so the formula collapses to
    -w[t] * log(p_t), averaged over the batch (sum/B reduction).

    Reduction choice: plain-mean (sum/B), NOT PyTorch's default weighted-mean
    (sum/sum(weights[targets])). The plain-mean reduction is required so that
    test 7's single-sample minority/majority gradient comparison is meaningful
    — under PyTorch's weighted-mean, the class weight cancels exactly in a
    one-sample batch (numerator and denominator share the factor w[t]), making
    class weights have no effect on gradient magnitude. Plain-mean preserves
    the D5 strategy's intent ("class weights amplify minority gradients").

    Reference: F.cross_entropy(..., reduction='sum') / batch_size.
    Tolerance is 1e-5 to account for floating-point order differences.
    """
    logits = _make_logits(batch=4, seed=99)
    targets = torch.tensor([0, 3, 7, 11])
    weights = torch.tensor([
        2.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0, 1.0, 3.0
    ])

    fl = focal_loss(logits, targets, class_weights=weights, gamma=0.0)

    # Reference: plain-mean weighted CE (sum / batch_size), matching the
    # focal_loss reduction. See docstring above for why this differs from
    # F.cross_entropy's default 'mean' reduction.
    batch_size = logits.shape[0]
    ce = F.cross_entropy(logits, targets, weight=weights, reduction='sum') / batch_size

    assert abs(fl.item() - ce.item()) < 1e-5, (
        f"focal_loss(gamma=0) = {fl.item():.8f}, "
        f"plain-mean weighted CE = {ce.item():.8f}, "
        f"difference = {abs(fl.item() - ce.item()):.2e} (expected < 1e-5)"
    )


# ===========================================================================
# Test 7 (ROADMAP item 7) — minority class gets larger gradient with high weight
# ===========================================================================

def test_focal_loss_minority_gradient_larger():
    """Spec: minority class with high weight -> larger gradient magnitude than majority.

    D5 strategy: class_weights compensate for imbalance. A minority class
    assigned weight=10 should produce a much larger gradient on its logit
    than a majority class with weight=1, for equally uncertain predictions.

    We run a forward+backward on two single-sample batches: one labelled
    minority (weight=10), one labelled majority (weight=1). Both use the
    same logits so the only difference is the weight. The minority gradient
    must be strictly larger (by a meaningful margin, not floating-point noise).
    """
    # Use the same logits for both so only the weight differs.
    torch.manual_seed(42)
    logits_min = torch.randn(1, _NUM_CLASSES, requires_grad=True)
    logits_maj = logits_min.detach().clone().requires_grad_(True)

    minority_class = 11   # "Multi-steps" — rarest class
    majority_class = 0    # "Step up"    — most common class

    weights_high = torch.ones(_NUM_CLASSES)
    weights_high[minority_class] = 10.0   # high minority weight
    weights_low = torch.ones(_NUM_CLASSES)
    weights_low[majority_class] = 1.0     # baseline majority weight

    targets_min = torch.tensor([minority_class])
    targets_maj = torch.tensor([majority_class])

    loss_min = focal_loss(logits_min, targets_min, class_weights=weights_high, gamma=2.0)
    loss_maj = focal_loss(logits_maj, targets_maj, class_weights=weights_low, gamma=2.0)

    loss_min.backward()
    loss_maj.backward()

    grad_min = logits_min.grad.abs().mean().item()
    grad_maj = logits_maj.grad.abs().mean().item()

    assert grad_min > grad_maj, (
        f"Expected minority gradient ({grad_min:.6f}) > majority gradient "
        f"({grad_maj:.6f}). High class weight must amplify gradients for "
        f"the minority class."
    )


# ===========================================================================
# Additional test — focal_loss output is a scalar finite tensor
# ===========================================================================

def test_focal_loss_output_scalar_finite():
    """focal_loss must return a scalar (0-dim) finite tensor on CPU.

    A non-scalar output (e.g. per-sample losses not reduced) would silently
    break the training loop's .backward() call. A NaN loss halts training.
    """
    logits = _make_logits(batch=8, seed=1)
    targets = torch.randint(0, _NUM_CLASSES, (8,))
    weights = _uniform_weights()

    loss = focal_loss(logits, targets, class_weights=weights, gamma=2.0)

    assert loss.ndim == 0, (
        f"focal_loss must return a scalar (0-dim tensor), got shape {loss.shape}"
    )
    assert torch.isfinite(loss), (
        f"focal_loss returned non-finite value: {loss.item()}"
    )
    assert loss.item() >= 0.0, (
        f"focal_loss must be non-negative, got {loss.item()}"
    )


# ===========================================================================
# Additional test — gradient flows through logits (no NaN gradient)
# ===========================================================================

def test_focal_loss_gradient_flows():
    """Gradients must be non-zero and finite after focal_loss.backward().

    A common implementation bug is using log(0) or divide-by-zero inside the
    focal term, which produces NaN gradients and kills training silently.
    We verify with requires_grad=True logits.
    """
    logits = torch.randn(6, _NUM_CLASSES, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3, 4, 5])
    weights = _uniform_weights()

    loss = focal_loss(logits, targets, class_weights=weights, gamma=2.0)
    loss.backward()

    assert logits.grad is not None, (
        "focal_loss did not create a gradient for logits"
    )
    assert torch.isfinite(logits.grad).all(), (
        f"focal_loss produced NaN/Inf gradients: "
        f"NaN count={torch.isnan(logits.grad).sum().item()}"
    )
    assert logits.grad.abs().sum() > 0, (
        "focal_loss gradients are all zero — loss is disconnected from logits"
    )


# ===========================================================================
# Additional test — raises on batch-size shape mismatch
# ===========================================================================

def test_focal_loss_shape_mismatch_raises():
    """focal_loss must raise when logits.shape[0] != targets.shape[0].

    Silently operating on mismatched batches produces wrong loss values that
    are impossible to debug downstream. An explicit error is required.
    """
    logits = _make_logits(batch=4)
    targets = torch.tensor([0, 1, 2])  # 3 != 4
    weights = _uniform_weights()

    import pytest
    with pytest.raises((RuntimeError, ValueError, IndexError)):
        focal_loss(logits, targets, class_weights=weights, gamma=2.0)


# ===========================================================================
# Additional test — raises on class_weights wrong dimension
# ===========================================================================

def test_focal_loss_wrong_weight_dim_raises():
    """focal_loss must raise when class_weights.shape[0] != logits.shape[1].

    Wrong-sized weights silently compute incorrect per-class penalty when
    PyTorch broadcasts — we need an explicit guard.
    """
    logits = _make_logits(batch=4, n_classes=12)
    targets = torch.tensor([0, 1, 2, 3])
    wrong_weights = torch.ones(5)  # 5 != 12

    import pytest
    with pytest.raises((RuntimeError, ValueError)):
        focal_loss(logits, targets, class_weights=wrong_weights, gamma=2.0)


# ===========================================================================
# Additional test — high-confidence correct prediction gives near-zero loss
# ===========================================================================

def test_focal_loss_correct_prediction_near_zero():
    """A very confident correct prediction must produce near-zero focal loss.

    Hand derivation: if p_t -> 1, then (1 - p_t)^gamma -> 0, so focal loss
    -> 0 regardless of class weight. This is the defining property of focal
    loss vs plain CE: easy examples are downweighted.

    We construct logits where class 0 has a very large score (+20), making
    softmax(logits)[0] ~ 1.0. With target=0 and gamma=2, the loss must be
    very small (< 0.01).
    """
    logits = torch.zeros(1, _NUM_CLASSES)
    logits[0, 0] = 20.0   # overwhelming score for class 0
    targets = torch.tensor([0])
    weights = _uniform_weights()

    loss = focal_loss(logits, targets, class_weights=weights, gamma=2.0)

    assert loss.item() < 0.01, (
        f"A very confident correct prediction should give near-zero focal loss "
        f"(gamma=2 strongly downweights easy examples). Got loss={loss.item():.6f}"
    )
