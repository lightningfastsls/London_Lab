"""Tests for usv_spectrogram.classifier.dann — Module 18.4 DANN components.

Written by test-architect BEFORE implementation exists. All tests will fail at
collection (ImportError) until dann.py is created. That is the expected TDD
red phase.

ROADMAP §18.4 test plan coverage:
  1. GradientReversal forward returns input unchanged (value-equal)
     ->  test_gradient_reversal_forward_identity
  2. GradientReversal backward returns gradient × −λ (λ ∈ {1.0, 0.5, 0.0})
     ->  test_gradient_reversal_backward_negates_by_lambda
  3. DomainHead 2-way output → logits shape (B, 2)
     ->  test_domain_head_output_shape
  4. LambdaSchedule.lambda_at(0) ≈ 0.0; lambda_at(total_epochs) ≈ 1.0; monotone
     ->  test_lambda_schedule_boundary_values_and_monotonicity
  5. ResNet18DANN.forward returns 3-tuple with shapes (B,12),(B,2),(B,512)
     ->  test_resnet18dann_forward_output_shapes
  6. ResNet18DANN with lambda_=0.0 in eval() produces identical class_logits
     ->  test_resnet18dann_deterministic_in_eval_mode
  9. Training smoke: adversarial loop 2 epochs, no NaN
     ->  test_dann_training_smoke_no_nan
  10. Training smoke: DANN macro-F1 within ±0.10 of no-DANN baseline
     ->  test_dann_collapse_detection_f1_within_band

Additional coverage (recurring gap patterns):
  - GradientReversal forward preserves shape             ->  (folded into test 1)
  - DomainHead with custom num_domains                   ->  test_domain_head_custom_num_domains
  - DomainHead output is finite (no NaN)                ->  test_domain_head_output_finite
  - LambdaSchedule rejects epoch > total_epochs         ->  test_lambda_schedule_epoch_clamp_or_cap
  - LambdaSchedule with gamma override                  ->  test_lambda_schedule_custom_gamma
  - ResNet18DANN features have shape (B, 512)            ->  (folded into test 5)
  - ResNet18DANN is an nn.Module                        ->  test_resnet18dann_is_nn_module
  - LambdaSchedule is a frozen dataclass                ->  test_lambda_schedule_frozen_dataclass
  - grad_reverse passthrough at lambda_=0               ->  test_grad_reverse_lambda_zero_no_gradient_effect

Total: 14 tests (8 from ROADMAP plan items 1-6,9,10; 6 additional gap-pattern tests)

Spec ambiguities resolved:
  - LambdaSchedule.lambda_at(epoch > total_epochs): spec says p=epoch/total_epochs
    but does not cap. We test clamping/non-crash behaviour — the test asserts only
    that the call does NOT raise (acceptable overflow into >1.0 is permitted).
  - ResNet18DANN image size: spec does not specify; we use 64×64 (fast on CPU).
    ResNet-18's global-average-pool means arbitrary spatial size works.
  - Training smoke (test 9): spec says "2 cages × 6 classes × 8 samples"; we use
    32×32 single-channel images promoted to 3-channel to keep the CPU test fast.
    The spec's 8 samples/class is respected.
  - F1 collapse band (test 10): spec says ±0.10; with tiny synthetic data both
    models will have noisy F1; we assert both are finite AND the difference ≤ 0.10.
    We do NOT weaken the band below 0.10.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Repo-root sys.path bootstrap (matches house style in test_model.py)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Import under test — will fail until dann.py exists (expected/correct).
# ---------------------------------------------------------------------------
from usv_spectrogram.classifier.dann import (  # noqa: E402
    GradientReversal,
    DomainHead,
    LambdaSchedule,
    ResNet18DANN,
    grad_reverse,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
_NUM_CLASSES = 12
_NUM_DOMAINS = 2
_FEATURE_DIM = 512   # ResNet-18 output feature dimension
_BATCH = 4
_IMG_H = 64          # small spatial size — ResNet-18's global pool handles any size
_IMG_W = 64
_SEED = 42


def _make_features(batch: int = _BATCH, dim: int = _FEATURE_DIM, seed: int = _SEED) -> torch.Tensor:
    """Return random (B, dim) float features on CPU."""
    torch.manual_seed(seed)
    return torch.randn(batch, dim)


def _make_images(batch: int = _BATCH, h: int = _IMG_H, w: int = _IMG_W, seed: int = _SEED) -> torch.Tensor:
    """Return random (B, 3, H, W) float images on CPU."""
    torch.manual_seed(seed)
    return torch.randn(batch, 3, h, w)


# ===========================================================================
# Test 1 (ROADMAP item 1) — GradientReversal forward is the identity
# ===========================================================================

def test_gradient_reversal_forward_identity():
    """Spec: GradientReversal.forward returns input unchanged (value-equal).

    The whole point of DANN is that the gradient reversal is ONLY seen
    during the backward pass. The forward pass must be a strict identity so
    that downstream layers receive the original features unmodified.
    """
    torch.manual_seed(_SEED)
    x = torch.randn(3, 8, requires_grad=True)
    out = grad_reverse(x, lambda_=1.0)

    assert out.shape == x.shape, (
        f"GradientReversal forward changed shape: {x.shape} -> {out.shape}"
    )
    # Values must be numerically identical — not just similar.
    assert torch.allclose(out, x, atol=0.0), (
        "GradientReversal forward must return values identical to input "
        f"(max diff = {(out - x).abs().max().item():.2e})"
    )


# ===========================================================================
# Test 2 (ROADMAP item 2) — GradientReversal backward reverses gradient × λ
# ===========================================================================

@pytest.mark.parametrize("lambda_val", [1.0, 0.5, 0.0])
def test_gradient_reversal_backward_negates_by_lambda(lambda_val: float):
    """Spec: GradientReversal backward returns grad_output.neg() * λ.

    Ganin 2015 equation: the domain classifier's loss gradient is reversed
    (multiplied by −λ) before reaching the encoder, encouraging it to produce
    features that *confuse* the domain classifier.

    We create a 1-D input tensor, push it through grad_reverse, then call
    .backward(upstream_grad) with a known upstream gradient. The input's
    .grad must equal −lambda_val × upstream_grad (element-wise).

    Hand-check for lambda_val=1.0, upstream=[1.0, 2.0, 3.0]:
      expected input.grad = [−1.0, −2.0, −3.0]
    """
    torch.manual_seed(_SEED)
    upstream_grad = torch.tensor([1.0, 2.0, 3.0])

    x = torch.tensor([0.5, -0.3, 1.2], requires_grad=True)
    out = grad_reverse(x, lambda_=lambda_val)

    # Use a scalar loss so .backward() works without a gradient argument.
    # We want upstream_grad to flow back, so construct: loss = dot(out, upstream_grad).
    # Then d(loss)/d(out) = upstream_grad, and d(loss)/d(x) = −lambda_val * upstream_grad.
    loss = (out * upstream_grad).sum()
    loss.backward()

    expected = -lambda_val * upstream_grad
    assert x.grad is not None, "GradientReversal did not create a gradient for x"
    assert torch.allclose(x.grad, expected, atol=1e-6), (
        f"GradientReversal backward with lambda_={lambda_val}: "
        f"expected grad={expected.tolist()}, got {x.grad.tolist()}"
    )


# ===========================================================================
# Test 3 (ROADMAP item 3) — DomainHead 2-way output shape is (B, 2)
# ===========================================================================

def test_domain_head_output_shape():
    """Spec: DomainHead(feature_dim, num_domains=2).forward(features, lambda_)
    returns logits of shape (B, 2).

    The domain head is the cage discriminator; with 2 cages (D4 decision)
    it must produce exactly 2 logits per sample.
    """
    head = DomainHead(feature_dim=_FEATURE_DIM, num_domains=2)
    head.eval()
    features = _make_features(batch=_BATCH)
    with torch.no_grad():
        logits = head(features, lambda_=1.0)
    assert logits.shape == (_BATCH, 2), (
        f"DomainHead output shape: expected ({_BATCH}, 2), got {tuple(logits.shape)}"
    )


# ===========================================================================
# Additional test — DomainHead with custom num_domains
# ===========================================================================

def test_domain_head_custom_num_domains():
    """DomainHead must honour an arbitrary num_domains argument.

    If per-recording cage granularity is ever enabled (50-100 domains),
    the same class must handle it. We test with num_domains=5.
    """
    head = DomainHead(feature_dim=_FEATURE_DIM, num_domains=5)
    head.eval()
    features = _make_features(batch=3)
    with torch.no_grad():
        logits = head(features, lambda_=0.5)
    assert logits.shape == (3, 5), (
        f"DomainHead with num_domains=5: expected (3, 5), got {tuple(logits.shape)}"
    )


# ===========================================================================
# Additional test — DomainHead output is finite
# ===========================================================================

def test_domain_head_output_finite():
    """DomainHead forward pass must produce finite logits (no NaN, no Inf).

    NaN logits in the domain head would propagate into the DANN loss and
    silently corrupt the encoder gradient during adversarial training.
    """
    head = DomainHead(feature_dim=_FEATURE_DIM, num_domains=2)
    head.eval()
    features = _make_features(batch=8)
    with torch.no_grad():
        logits = head(features, lambda_=1.0)
    assert torch.isfinite(logits).all(), (
        f"DomainHead produced non-finite logits: "
        f"NaN={torch.isnan(logits).sum().item()}, "
        f"Inf={torch.isinf(logits).sum().item()}"
    )


# ===========================================================================
# Test 4 (ROADMAP item 4) — LambdaSchedule boundary values and monotonicity
# ===========================================================================

def test_lambda_schedule_boundary_values_and_monotonicity():
    """Spec: LambdaSchedule uses Ganin 2015 formula: 2/(1+exp(-γ·p)) - 1.

    Boundary values (hand-computed):
      epoch=0:            p=0.0, λ = 2/(1+exp(0)) - 1 = 2/2 - 1 = 0.0  (exact)
      epoch=total_epochs: p=1.0, λ = 2/(1+exp(-10)) - 1 ≈ 0.9999   (within 1e-3 of 1.0)

    Monotonicity: λ must be non-decreasing across all epochs 0..total_epochs
    (a strictly increasing sigmoid — no epoch should produce a lower λ than
    its predecessor).

    Using total_epochs=20, gamma=10.0 (defaults) to match spec.
    """
    schedule = LambdaSchedule(total_epochs=20)

    # Exact boundary: epoch 0
    lambda_at_0 = schedule.lambda_at(0)
    assert abs(lambda_at_0 - 0.0) < 1e-9, (
        f"LambdaSchedule.lambda_at(0) must equal 0.0 exactly, got {lambda_at_0}"
    )

    # Near-1.0 boundary: epoch=total_epochs (p=1)
    lambda_at_end = schedule.lambda_at(20)
    assert abs(lambda_at_end - 1.0) < 1e-3, (
        f"LambdaSchedule.lambda_at(total_epochs) must be within 1e-3 of 1.0, "
        f"got {lambda_at_end}"
    )

    # Strict monotonicity across all epochs
    values = [schedule.lambda_at(e) for e in range(21)]
    for i in range(len(values) - 1):
        assert values[i] <= values[i + 1], (
            f"LambdaSchedule is not monotone: lambda_at({i})={values[i]:.6f} "
            f"> lambda_at({i+1})={values[i+1]:.6f}"
        )


# ===========================================================================
# Additional test — LambdaSchedule honours custom gamma
# ===========================================================================

def test_lambda_schedule_custom_gamma():
    """LambdaSchedule with custom gamma must apply the correct formula.

    With gamma=1.0, total_epochs=10, epoch=5:
      p = 5/10 = 0.5
      λ = 2/(1+exp(-1.0*0.5)) - 1 = 2/(1+exp(-0.5)) - 1
      exp(-0.5) ≈ 0.60653
      λ ≈ 2/1.60653 - 1 ≈ 1.24496 - 1 = 0.24496...
    Tolerance: 1e-5.
    """
    schedule = LambdaSchedule(total_epochs=10, gamma=1.0)
    result = schedule.lambda_at(5)
    expected = 2.0 / (1.0 + math.exp(-1.0 * 0.5)) - 1.0
    assert abs(result - expected) < 1e-5, (
        f"LambdaSchedule(gamma=1.0).lambda_at(5) expected {expected:.6f}, got {result:.6f}"
    )


# ===========================================================================
# Additional test — LambdaSchedule is a frozen dataclass
# ===========================================================================

def test_lambda_schedule_frozen_dataclass():
    """LambdaSchedule must be a frozen dataclass (immutable after construction).

    Schedules should be treated as value objects — accidental mutation of
    total_epochs mid-training would silently corrupt the λ curve.
    """
    schedule = LambdaSchedule(total_epochs=50)
    # Frozen dataclasses raise FrozenInstanceError (a subclass of TypeError/AttributeError)
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError, AttributeError)):
        schedule.total_epochs = 99  # type: ignore[misc]


# ===========================================================================
# Additional test — LambdaSchedule epoch > total_epochs does not crash
# ===========================================================================

def test_lambda_schedule_epoch_clamp_or_cap():
    """LambdaSchedule.lambda_at(epoch > total_epochs) must not raise.

    The spec formula p=epoch/total_epochs produces p>1 for epoch>total_epochs,
    which is valid (sigmoid approaches 1 asymptotically). We only assert no
    exception is raised and the result is a finite float. The exact value
    above 1.0 is implementation-specific and acceptable.
    """
    schedule = LambdaSchedule(total_epochs=10)
    result = schedule.lambda_at(15)   # epoch beyond training
    assert math.isfinite(result), (
        f"lambda_at(epoch > total_epochs) must return a finite float, got {result}"
    )


# ===========================================================================
# Test 5 (ROADMAP item 5) — ResNet18DANN forward returns correct shapes
# ===========================================================================

def test_resnet18dann_forward_output_shapes():
    """Spec: ResNet18DANN.forward(x, lambda_) -> (class_logits, domain_logits, features)
    with shapes (B,12), (B,2), (B,512).

    ResNet-18 produces 512-dim features before the final FC. The class head
    maps to 12 (Grimsley taxonomy), the domain head to 2 (D4: 2 cages).
    """
    model = ResNet18DANN(num_classes=12, num_domains=2, pretrained=False)
    model.eval()
    x = _make_images(batch=_BATCH)
    with torch.no_grad():
        class_logits, domain_logits, features = model(x, lambda_=0.5)

    assert class_logits.shape == (_BATCH, 12), (
        f"class_logits shape: expected ({_BATCH}, 12), got {tuple(class_logits.shape)}"
    )
    assert domain_logits.shape == (_BATCH, 2), (
        f"domain_logits shape: expected ({_BATCH}, 2), got {tuple(domain_logits.shape)}"
    )
    assert features.shape == (_BATCH, 512), (
        f"features shape: expected ({_BATCH}, 512), got {tuple(features.shape)}"
    )


# ===========================================================================
# Additional test — ResNet18DANN is an nn.Module
# ===========================================================================

def test_resnet18dann_is_nn_module():
    """ResNet18DANN must be an nn.Module subclass.

    This is required for .parameters(), .state_dict(), .to(device), etc.
    to work in the training loop.
    """
    model = ResNet18DANN(pretrained=False)
    assert isinstance(model, nn.Module), (
        f"ResNet18DANN must be an nn.Module, got {type(model).__name__}"
    )


# ===========================================================================
# Test 6 (ROADMAP item 6) — ResNet18DANN is deterministic in eval() mode
# ===========================================================================

def test_resnet18dann_deterministic_in_eval_mode():
    """Spec: ResNet18DANN with lambda_=0.0 in eval() mode produces identical
    class_logits across two forward calls.

    With lambda_=0.0 the gradient reversal scales gradients by 0 (not the
    forward), but more importantly Dropout in DomainHead must be disabled
    in eval() mode. Two forward passes with the same input must produce
    bit-for-bit identical class_logits.

    Note: lambda_=0.0 is the initial state at the start of DANN training
    (epoch 0 → p=0 → λ=0). This is the most commonly tested checkpoint.
    """
    torch.manual_seed(_SEED)
    model = ResNet18DANN(num_classes=12, num_domains=2, pretrained=False)
    model.eval()   # disables BatchNorm running stats updates and Dropout

    x = _make_images(batch=2)
    with torch.no_grad():
        logits_1, _, _ = model(x, lambda_=0.0)
        logits_2, _, _ = model(x, lambda_=0.0)

    assert torch.equal(logits_1, logits_2), (
        "ResNet18DANN in eval() mode with lambda_=0.0 produced different "
        "class_logits on two identical forward passes — Dropout or BatchNorm "
        "is not properly disabled in eval mode."
    )


# ===========================================================================
# Test 9 (ROADMAP item 9) — adversarial training smoke: 2 epochs, no NaN
# ===========================================================================

def test_dann_training_smoke_no_nan():
    """Spec: 2 cages × 6 classes × 8 samples trains 2 epochs without NaN.

    We implement a minimal adversarial training step directly on
    ResNet18DANN + cross-entropy (class) + cross-entropy (domain). This
    tests that the combined forward/backward with gradient reversal does not
    produce NaN losses or NaN parameters within 2 epochs.

    Spec parameters: 2 cages, 6 classes, 8 samples/class (96 samples total).
    Images: 32×32 (faster than 64×64 for smoke speed). CPU only.
    pretrained=False.
    """
    torch.manual_seed(_SEED)
    n_cages = 2
    n_classes = 6
    n_per_class = 8
    n_total = n_classes * n_per_class   # 48 samples
    img_h, img_w = 32, 32

    images = torch.randn(n_total, 3, img_h, img_w)
    # class labels: 0..5 repeated 8 times
    class_labels = torch.arange(n_classes).repeat(n_per_class)
    # cage labels: alternate cage assignment (half in cage 0, half in cage 1)
    cage_labels = torch.zeros(n_total, dtype=torch.long)
    cage_labels[n_total // 2:] = 1

    dataset = TensorDataset(images, class_labels, cage_labels)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = ResNet18DANN(num_classes=n_classes, num_domains=n_cages, pretrained=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    schedule = LambdaSchedule(total_epochs=2)

    for epoch in range(2):
        lam = schedule.lambda_at(epoch)
        model.train()
        for imgs, cls_tgt, dom_tgt in loader:
            optimizer.zero_grad()
            class_logits, domain_logits, _ = model(imgs, lambda_=lam)
            loss_cls = nn.functional.cross_entropy(class_logits, cls_tgt)
            loss_dom = nn.functional.cross_entropy(domain_logits, dom_tgt)
            loss = loss_cls + loss_dom
            loss.backward()
            optimizer.step()

            assert torch.isfinite(loss_cls), (
                f"Epoch {epoch}: class loss is NaN/Inf ({loss_cls.item()})"
            )
            assert torch.isfinite(loss_dom), (
                f"Epoch {epoch}: domain loss is NaN/Inf ({loss_dom.item()})"
            )

    # Check no NaN in parameters after training
    for name, param in model.named_parameters():
        assert torch.isfinite(param).all(), (
            f"Parameter '{name}' contains NaN/Inf after 2 training epochs"
        )


# ===========================================================================
# Test 10 (ROADMAP item 10) — DANN macro-F1 within ±0.10 of no-DANN baseline
# ===========================================================================

def test_dann_collapse_detection_f1_within_band():
    """Spec: v2-style macro-F1 within ±0.10 of v1-style baseline (collapse detection).

    We train two models on the same tiny synthetic dataset (fixed seed):
      - Baseline: ResNet18DANN with lambda_=0.0 throughout (effectively v1)
      - DANN: ResNet18DANN with the full LambdaSchedule

    Both models are evaluated on the same held-out set. The absolute
    difference in macro-F1 must be ≤ 0.10. This catches encoder collapse:
    if DANN's λ is too aggressive, its F1 will collapse to near 0 while
    the baseline's remains higher.

    Seed is fixed for reproducibility. We assert:
      1. Both F1 values are finite (neither model crashed).
      2. |f1_dann - f1_base| ≤ 0.10 (spec band, not weakened).
    """
    from sklearn.metrics import f1_score

    torch.manual_seed(2024)
    n_classes = 6
    n_cages = 2
    n_train = n_classes * 8    # 48 train samples
    n_val = n_classes * 4      # 24 val samples
    img_h, img_w = 32, 32

    def _make_loader(n: int, seed: int, batch: int = 16) -> DataLoader:
        torch.manual_seed(seed)
        imgs = torch.randn(n, 3, img_h, img_w)
        # Give the images a weak class-discriminative signal (add per-class mean)
        cls_lbl = torch.arange(n_classes).repeat(n // n_classes)
        for c in range(n_classes):
            imgs[cls_lbl == c] += c * 0.1   # small but consistent offset
        cage_lbl = torch.zeros(n, dtype=torch.long)
        cage_lbl[n // 2:] = 1
        return DataLoader(TensorDataset(imgs, cls_lbl, cage_lbl), batch_size=batch)

    train_loader = _make_loader(n_train, seed=100)
    val_loader = _make_loader(n_val, seed=200)

    def _train_and_eval(use_dann: bool) -> float:
        """Train for 5 epochs, return val macro-F1."""
        torch.manual_seed(_SEED)
        model = ResNet18DANN(num_classes=n_classes, num_domains=n_cages, pretrained=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = LambdaSchedule(total_epochs=5)

        for epoch in range(5):
            lam = sched.lambda_at(epoch) if use_dann else 0.0
            model.train()
            for imgs, cls_tgt, dom_tgt in train_loader:
                optimizer.zero_grad()
                class_logits, domain_logits, _ = model(imgs, lambda_=lam)
                loss = nn.functional.cross_entropy(class_logits, cls_tgt)
                if use_dann:
                    loss = loss + nn.functional.cross_entropy(domain_logits, dom_tgt)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for imgs, cls_tgt, _ in val_loader:
                class_logits, _, _ = model(imgs, lambda_=0.0)
                preds = class_logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_true.extend(cls_tgt.cpu().numpy().tolist())

        return f1_score(all_true, all_preds, average="macro", zero_division=0)

    f1_base = _train_and_eval(use_dann=False)
    f1_dann = _train_and_eval(use_dann=True)

    assert math.isfinite(f1_base), f"Baseline F1 is not finite: {f1_base}"
    assert math.isfinite(f1_dann), f"DANN F1 is not finite: {f1_dann}"

    diff = abs(f1_dann - f1_base)
    assert diff <= 0.10, (
        f"DANN macro-F1 ({f1_dann:.4f}) differs from baseline ({f1_base:.4f}) "
        f"by {diff:.4f}, which exceeds the ±0.10 collapse-detection band. "
        f"This indicates possible encoder collapse — check λ schedule."
    )


# ===========================================================================
# Additional test — grad_reverse with lambda_=0.0 has no gradient effect
# ===========================================================================

def test_grad_reverse_lambda_zero_no_gradient_effect():
    """With lambda_=0.0, grad_reverse must pass gradients through unchanged (×0).

    At epoch 0, p=0 → λ=0. The domain head's gradient contribution is zeroed
    out completely — the encoder receives no domain signal yet. This is the
    warm-up phase described in Ganin 2015.

    Specifically: input.grad should be exactly zero (−0.0 × upstream = 0.0).
    """
    x = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
    upstream_grad = torch.tensor([4.0, 5.0, 6.0])

    out = grad_reverse(x, lambda_=0.0)
    loss = (out * upstream_grad).sum()
    loss.backward()

    expected = torch.zeros(3)
    assert torch.allclose(x.grad, expected, atol=1e-9), (
        f"grad_reverse(lambda_=0.0) should produce zero input gradient, "
        f"got {x.grad.tolist()}"
    )
