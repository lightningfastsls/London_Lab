"""Adversarial / edge-case tests for usv_spectrogram.classifier.dann — Module 18.4.

Added by test-hardener after master-reviewer approval of Module 18.4.

Gaps targeted (not covered by the original 14 tests in test_dann.py):

  A. grad_reverse sign behaviour with negative and very large lambda.
  B. GradientReversal forward preserves dtype (float16, float64).
  C. Gradient flows to an UPSTREAM parameter through grad_reverse (not just
     the leaf tensor passed directly).
  D. LambdaSchedule: total_epochs=1 (no div-by-zero), gamma=0 (lambda≡0 for
     all p), epoch>total_epochs stays finite and in a reasonable range.
  E. DomainHead: end-to-end gradient sign flip — wiring a tiny encoder through
     grad_reverse to DomainHead and checking the encoder-param grad sign flips
     compared to lambda_=0 (positive gradient vs negative).
  F. ResNet18DANN: non-default num_classes/num_domains produce correct shapes.
  G. ResNet18DANN: features.requires_grad in train mode (gradients flow back).
  H. ResNet18DANN: domain_logits are NOT identical across two forward calls in
     train() mode (Dropout in DomainHead is live in train mode).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Repo-root sys.path bootstrap (matches house style)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

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
_SEED = 42
_FEATURE_DIM = 512


# ===========================================================================
# A. grad_reverse sign behaviour with edge-case lambda values
# ===========================================================================

def test_grad_reverse_negative_lambda_flips_sign():
    """grad_reverse with lambda_ = -1.0 should ADD the gradient (double flip).

    The backward formula is: grad_input = grad_output.neg() * lambda_
    With lambda_ = -1.0: grad_input = -upstream * (-1) = +upstream.
    This is the identity gradient — negative lambda un-reverses the reversal.
    """
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    upstream = torch.tensor([4.0, 5.0, 6.0])
    out = grad_reverse(x, lambda_=-1.0)
    loss = (out * upstream).sum()
    loss.backward()

    # With lambda_=-1.0: grad = -upstream * (-1) = upstream
    expected = upstream
    assert x.grad is not None
    assert torch.allclose(x.grad, expected, atol=1e-6), (
        f"grad_reverse(lambda_=-1.0): expected grad={expected.tolist()}, "
        f"got {x.grad.tolist()}"
    )


def test_grad_reverse_large_lambda_scales_gradient():
    """grad_reverse with lambda_ = 100.0 should multiply the reversed gradient by 100.

    Large lambda is the high-adversarial-pressure scenario. The formula still
    holds: grad_input = -upstream * 100.
    """
    torch.manual_seed(_SEED)
    x = torch.tensor([1.0, -1.0, 2.0], requires_grad=True)
    upstream = torch.ones(3)
    out = grad_reverse(x, lambda_=100.0)
    loss = (out * upstream).sum()
    loss.backward()

    expected = -100.0 * upstream
    assert x.grad is not None
    assert torch.allclose(x.grad, expected, atol=1e-4), (
        f"grad_reverse(lambda_=100.0): expected {expected.tolist()}, got {x.grad.tolist()}"
    )


# ===========================================================================
# B. GradientReversal forward preserves dtype
# ===========================================================================

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gradient_reversal_preserves_dtype(dtype: torch.dtype):
    """GradientReversal forward must preserve the input tensor dtype.

    float16 is excluded from parametrize because some PyTorch builds do not
    support float16 autograd on CPU; float32 and float64 are always supported.
    If dtype is corrupted to float32, downstream BN layers can fail silently
    with wrong numerics.
    """
    torch.manual_seed(_SEED)
    x = torch.randn(4, 8, dtype=dtype, requires_grad=True)
    out = grad_reverse(x, lambda_=1.0)
    assert out.dtype == dtype, (
        f"GradientReversal changed dtype from {dtype} to {out.dtype}"
    )


# ===========================================================================
# C. Gradient flows to an upstream parameter through grad_reverse
# ===========================================================================

def test_gradient_reversal_flows_to_upstream_parameter():
    """grad_reverse must propagate gradients to upstream parameters, not only
    to the direct input tensor.

    We build: Linear (upstream_param) → relu → grad_reverse(λ=1) → sum → loss.
    The upstream Linear's weight must receive a non-zero gradient — confirming
    that the reversal does not break the computational graph for parameters
    sitting earlier in the chain.
    """
    torch.manual_seed(_SEED)
    linear = nn.Linear(4, 4, bias=False)
    x = torch.randn(2, 4)

    h = torch.relu(linear(x))
    h_reversed = grad_reverse(h, lambda_=1.0)
    loss = h_reversed.sum()
    loss.backward()

    assert linear.weight.grad is not None, (
        "GradientReversal did not propagate gradient to upstream Linear weight"
    )
    assert not torch.all(linear.weight.grad == 0), (
        "GradientReversal propagated all-zero gradient to upstream Linear weight"
    )


# ===========================================================================
# D. LambdaSchedule edge-case inputs
# ===========================================================================

def test_lambda_schedule_total_epochs_one_no_div_by_zero():
    """LambdaSchedule(total_epochs=1) must not raise ZeroDivisionError.

    The implementation guards with max(1, total_epochs); we verify this also
    works for the boundary case total_epochs=1, and that lambda_at(0) and
    lambda_at(1) are both finite and in [0, 1].
    """
    schedule = LambdaSchedule(total_epochs=1)
    lam_0 = schedule.lambda_at(0)
    lam_1 = schedule.lambda_at(1)

    assert math.isfinite(lam_0), f"LambdaSchedule(1).lambda_at(0) is not finite: {lam_0}"
    assert math.isfinite(lam_1), f"LambdaSchedule(1).lambda_at(1) is not finite: {lam_1}"
    assert 0.0 <= lam_0 <= 1.0, f"lambda_at(0) out of [0,1]: {lam_0}"
    assert 0.0 <= lam_1 <= 1.0, f"lambda_at(1) out of [0,1]: {lam_1}"


def test_lambda_schedule_gamma_zero_lambda_equals_zero():
    """LambdaSchedule with gamma=0.0 should produce lambda=0 for all epochs.

    With gamma=0: 2/(1+exp(0)) - 1 = 2/2 - 1 = 0.0 regardless of p.
    This is a degenerate schedule where the domain signal is permanently off —
    valid for debugging and as a no-adversarial baseline.
    """
    schedule = LambdaSchedule(total_epochs=10, gamma=0.0)
    for epoch in [0, 1, 5, 10, 15]:
        lam = schedule.lambda_at(epoch)
        assert abs(lam - 0.0) < 1e-9, (
            f"LambdaSchedule(gamma=0).lambda_at({epoch}) expected 0.0, got {lam}"
        )


def test_lambda_schedule_epoch_beyond_total_bounded():
    """LambdaSchedule.lambda_at(epoch >> total_epochs) must stay finite and ≤ 2.

    The spec permits p>1 (no cap), but sigmoid(x) → 1 as x→∞, so the value
    must approach 1.0 from below and never exceed 2/(1+0)-1 = 1.0 in the limit.
    We test with epoch=10×total_epochs and verify the result is finite and < 2.
    """
    schedule = LambdaSchedule(total_epochs=10, gamma=10.0)
    lam = schedule.lambda_at(100)  # epoch = 10× total_epochs
    assert math.isfinite(lam), f"lambda_at(100) with total_epochs=10 is not finite: {lam}"
    assert lam < 2.0, (
        f"lambda_at(100) = {lam:.6f} exceeds the theoretical sigmoid ceiling of 1.0 "
        f"(max possible value is 1.0 as p→∞; < 2.0 is the loose sanity bound)"
    )


# ===========================================================================
# E. DomainHead end-to-end gradient sign reversal
# ===========================================================================

def test_domain_head_gradient_reverses_encoder_gradient():
    """Wiring encoder → grad_reverse → DomainHead must flip the sign of the
    encoder parameter gradient relative to the lambda_=0 (no-reversal) case.

    Setup: a tiny Linear encoder (1 parameter matrix, no bias).
    Forward:  features = encoder(x)
    Domain:   logits   = DomainHead(features, lambda_=lambda_val)
    Loss:     cross-entropy(logits, target)

    We compare the sign of encoder.weight.grad:
      - lambda_=0.0: grad_reverse zeroes the gradient → encoder.weight.grad = 0
      - lambda_=1.0: gradient is reversed → encoder.weight.grad ≠ 0 and has
        opposite sign to the natural (lambda_=-1.0) gradient.

    Concretely, we compare lambda_=1.0 vs lambda_=-1.0 (which passes gradient
    unchanged multiplied by +1 due to double negation) and verify the signs are
    opposite element-wise for at least one parameter.
    """
    torch.manual_seed(_SEED)
    feature_dim = 8
    encoder = nn.Linear(feature_dim, feature_dim, bias=False)
    domain_head = DomainHead(feature_dim=feature_dim, num_domains=2)
    x = torch.randn(4, feature_dim)
    target = torch.zeros(4, dtype=torch.long)  # all cage 0

    def _get_encoder_grad(lambda_val: float) -> torch.Tensor:
        encoder.zero_grad()
        domain_head.zero_grad()
        features = encoder(x)
        logits = domain_head(features, lambda_=lambda_val)
        loss = nn.functional.cross_entropy(logits, target)
        loss.backward()
        return encoder.weight.grad.clone()

    grad_positive = _get_encoder_grad(lambda_val=1.0)   # reversed gradient
    grad_negative = _get_encoder_grad(lambda_val=-1.0)  # un-reversed (double flip = pass-through)

    # With lambda_=1.0  : encoder.weight.grad = -1 × (natural gradient)
    # With lambda_=-1.0 : encoder.weight.grad = +1 × (natural gradient)
    # So they must have opposite signs on most elements.
    # We check: grad_positive and grad_negative are (a) both non-zero, and
    # (b) their element-wise product is ≤ 0 on the majority of entries.
    assert grad_positive is not None and not torch.all(grad_positive == 0), (
        "lambda_=1.0 produced zero encoder gradient; gradient reversal not working"
    )
    assert grad_negative is not None and not torch.all(grad_negative == 0), (
        "lambda_=-1.0 produced zero encoder gradient; check test setup"
    )
    product = grad_positive * grad_negative
    fraction_opposite = (product < 0).float().mean().item()
    assert fraction_opposite > 0.5, (
        f"Expected majority of encoder weight gradients to flip sign between "
        f"lambda_=1.0 and lambda_=-1.0, but only {fraction_opposite:.2%} had opposite signs. "
        f"The gradient-reversal layer may not be connected to the encoder path."
    )


# ===========================================================================
# F. ResNet18DANN: non-default num_classes / num_domains
# ===========================================================================

@pytest.mark.parametrize("num_classes,num_domains", [
    (5, 3),
    (1, 2),
    (12, 10),
])
def test_resnet18dann_custom_classes_and_domains(num_classes: int, num_domains: int):
    """ResNet18DANN must honour arbitrary num_classes and num_domains.

    The class head shape is (feature_dim, num_classes) and the domain head
    output is (B, num_domains). Both shapes must be respected for any valid
    positive-integer combination.
    """
    model = ResNet18DANN(num_classes=num_classes, num_domains=num_domains, pretrained=False)
    model.eval()
    torch.manual_seed(_SEED)
    x = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        class_logits, domain_logits, features = model(x, lambda_=0.5)

    assert class_logits.shape == (3, num_classes), (
        f"class_logits shape: expected (3, {num_classes}), got {tuple(class_logits.shape)}"
    )
    assert domain_logits.shape == (3, num_domains), (
        f"domain_logits shape: expected (3, {num_domains}), got {tuple(domain_logits.shape)}"
    )
    assert features.shape[0] == 3 and features.shape[1] == _FEATURE_DIM, (
        f"features shape: expected (3, 512), got {tuple(features.shape)}"
    )


# ===========================================================================
# G. ResNet18DANN: features have requires_grad=True in train mode
# ===========================================================================

def test_resnet18dann_features_require_grad_in_train_mode():
    """In train mode, the features returned by ResNet18DANN must be part of the
    autograd graph (requires_grad=True).

    This is required for the DANN loss to back-propagate through both heads into
    the encoder. If features.requires_grad is False in train mode, the domain
    head's loss gradient is silently detached and the encoder never learns
    cage invariance.
    """
    model = ResNet18DANN(num_classes=12, num_domains=2, pretrained=False)
    model.train()
    torch.manual_seed(_SEED)
    x = torch.randn(2, 3, 32, 32)
    _, _, features = model(x, lambda_=0.5)
    assert features.requires_grad, (
        "ResNet18DANN features do not have requires_grad=True in train mode. "
        "This would silently block gradient flow through the domain head."
    )


# ===========================================================================
# H. ResNet18DANN: domain_logits vary in train mode (Dropout active)
# ===========================================================================

def test_resnet18dann_domain_logits_vary_in_train_mode():
    """In train() mode, DomainHead has Dropout(0.5) active. Two forward passes
    with the same input should produce DIFFERENT domain_logits.

    If they are identical it means Dropout is not active in train mode — a bug
    that removes the regularisation benefit of the dropout layer.

    Note: the probability that two random Dropout masks are identical over 256
    neurons is astronomically small (2^{-256}), so any equality here is a real bug.
    """
    model = ResNet18DANN(num_classes=12, num_domains=2, pretrained=False)
    model.train()
    torch.manual_seed(_SEED)
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        _, domain_logits_1, _ = model(x, lambda_=0.5)
        _, domain_logits_2, _ = model(x, lambda_=0.5)

    assert not torch.equal(domain_logits_1, domain_logits_2), (
        "ResNet18DANN domain_logits are identical across two forward passes in "
        "train() mode. Dropout in DomainHead should create different masks each "
        "call — if they match, Dropout is not active."
    )


# ===========================================================================
# I. ResNet18DANN: class_logits are deterministic in eval mode
# ===========================================================================

def test_resnet18dann_class_logits_deterministic_in_eval():
    """class_logits must be identical across forward passes in eval() mode.

    eval() disables Dropout in the DomainHead but class_logits go through
    the encoder + class_head only — no Dropout there. Two passes should
    produce bit-for-bit identical class_logits.

    This is a targeted restatement of the original test_resnet18dann_deterministic_in_eval_mode
    but focuses specifically on class_logits being unaffected by domain-head Dropout.
    """
    model = ResNet18DANN(num_classes=12, num_domains=2, pretrained=False)
    model.eval()
    torch.manual_seed(_SEED)
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        class_logits_1, _, _ = model(x, lambda_=1.0)
        class_logits_2, _, _ = model(x, lambda_=1.0)

    assert torch.equal(class_logits_1, class_logits_2), (
        "class_logits changed across two eval() forward passes — Dropout or "
        "BatchNorm is not properly disabled."
    )


# ===========================================================================
# J. LambdaSchedule: strictly increasing (not just non-decreasing)
# ===========================================================================

def test_lambda_schedule_strictly_increasing_for_default_gamma():
    """With gamma=10.0 (strict positive), lambda must be *strictly* increasing
    for consecutive integer epochs up to total_epochs.

    The sigmoid formula with gamma>0 and strictly increasing p yields a strictly
    increasing lambda sequence. A non-strictly-increasing (flat) sequence would
    indicate a rounding collapse where small p differences vanish in float64.
    """
    schedule = LambdaSchedule(total_epochs=20, gamma=10.0)
    values = [schedule.lambda_at(e) for e in range(21)]
    for i in range(len(values) - 1):
        assert values[i] < values[i + 1], (
            f"LambdaSchedule(gamma=10) is not strictly increasing at epoch {i}: "
            f"lambda_at({i})={values[i]:.8f} >= lambda_at({i+1})={values[i+1]:.8f}"
        )


# ===========================================================================
# K. grad_reverse: forward value equality with lambda_ ≠ 1.0 (sanity)
# ===========================================================================

@pytest.mark.parametrize("lambda_val", [-1.0, 0.0, 0.3, 2.5, 100.0])
def test_grad_reverse_forward_identity_various_lambdas(lambda_val: float):
    """grad_reverse forward must be the identity regardless of lambda_ value.

    Forward pass is ALWAYS x — lambda_ only affects the backward pass.
    This is critical: if someone passes an unusually large or negative lambda_
    the forward tensor values must not change.
    """
    torch.manual_seed(_SEED)
    x = torch.randn(5, 8, requires_grad=True)
    out = grad_reverse(x, lambda_=lambda_val)
    assert torch.allclose(out, x, atol=0.0), (
        f"grad_reverse(lambda_={lambda_val}) changed forward values: "
        f"max diff = {(out - x).abs().max().item():.2e}"
    )
