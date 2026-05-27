"""Module 18.4 — DANN (Domain-Adversarial Neural Network) components.

ROADMAP §18.4. Adds a gradient-reversal cage discriminator on top of the
Module 18.3 ResNet-18 classifier so the shared encoder learns a
*cage-invariant* representation: good at the 12-class Grimsley syllable task,
bad at guessing which recording cage (D4: ``lab_131204`` vs ``vocalmat``) a
patch came from.

Mechanism (Ganin & Lempitsky 2015, arXiv:1409.7495):
  - The encoder feeds two heads — a 12-way class head and a 2-way domain head.
  - A *gradient-reversal layer* (GRL) sits between encoder and domain head.
    Forward it is the identity; backward it multiplies the gradient by −λ.
  - The domain head minimises its own cage-classification loss (gets *good* at
    cage discrimination), but the reversed gradient pushes the encoder in the
    opposite direction (gets *bad* at it) → cage-invariant features.

The central failure mode is **encoder collapse**: if λ ramps too aggressively
the encoder can find a trivially cage-invariant representation that also throws
away the syllable signal. The λ schedule ``2/(1+e^{−γp})−1`` (slow warm-up from
0) and the downstream collapse tripwire (``syllable F1 ≥ v1 − 0.05``) exist to
catch exactly that.

This module is purely additive — the Module 18.3 files (``model.py``,
``training.py``, ``losses.py``, ``augmentation.py``) are untouched. The shared
encoder here is built from the SAME timm ResNet-18 backbone as ``model.py`` so
v1 weights can be warm-started into it (see ``scripts/train_lab_classifier_v2.py``).

Reference: Ganin & Lempitsky 2015 ICML, "Unsupervised Domain Adaptation by
Backpropagation". He et al. 2015 (arXiv:1512.03385) for the ResNet-18 backbone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import timm
import torch
import torch.nn as nn

# ResNet-18 emits a 512-dim feature vector after global average pooling.
RESNET18_FEATURE_DIM = 512


class GradientReversal(torch.autograd.Function):
    """Gradient-reversal layer (Ganin 2015).

    Forward pass is the identity. Backward pass multiplies the incoming
    gradient by ``-lambda_``. λ is stashed on ``ctx`` during forward so the
    backward pass can apply it; it is a plain Python float, not a tensor, so
    the second element of the backward tuple (its gradient) is ``None``.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:  # noqa: D401
        ctx.lambda_ = lambda_
        # view_as is a cheap identity that still routes through autograd.
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # d(loss)/d(x) = -lambda_ * d(loss)/d(output). Second return is the
        # gradient w.r.t. lambda_ (a non-tensor constant) → None.
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    """Apply the gradient-reversal layer to ``x`` with reversal strength ``lambda_``."""
    return GradientReversal.apply(x, lambda_)


class DomainHead(nn.Module):
    """Cage discriminator on top of the gradient-reversal layer.

    D4 commits to 2-cage granularity (``lab_131204`` vs ``vocalmat``), the
    minimum-aggressiveness option. ``num_domains`` is parameterised so the more
    aggressive per-recording option (50–100 cages) can be enabled later without
    a code change.

    The MLP (Linear→ReLU→Dropout→Linear) is applied *after* the GRL, so its
    gradients reach the encoder already reversed.
    """

    def __init__(self, feature_dim: int, num_domains: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_domains),
        )

    def forward(self, features: torch.Tensor, lambda_: float) -> torch.Tensor:
        return self.net(grad_reverse(features, lambda_))


@dataclass(frozen=True)
class LambdaSchedule:
    """Ganin 2015 λ schedule: ``λ(p) = 2 / (1 + exp(-γ·p)) − 1``, ``p = epoch/total_epochs``.

    λ ramps smoothly from 0 (epoch 0 → encoder sees no domain signal, pure
    warm-up) toward ~1 (epoch ``total_epochs`` → full adversarial pressure).
    Frozen so a schedule is an immutable value object — accidentally mutating
    ``total_epochs`` mid-training would silently bend the λ curve.
    """

    total_epochs: int
    gamma: float = 10.0

    def lambda_at(self, epoch: int) -> float:
        p = epoch / max(1, self.total_epochs)
        return 2.0 / (1.0 + math.exp(-self.gamma * p)) - 1.0


class ResNet18DANN(nn.Module):
    """Shared ResNet-18 encoder + 12-class syllable head + 2-class cage head.

    The encoder is a timm ResNet-18 with the classifier stripped
    (``num_classes=0`` → forward returns the 512-dim pooled feature vector),
    identical in architecture to ``model.build_resnet18_classifier`` so v1
    weights warm-start cleanly. ``forward`` returns a 3-tuple
    ``(class_logits, domain_logits, features)``; the caller passes the current
    λ so the domain head's gradient is reversed by the right amount.
    """

    def __init__(
        self,
        num_classes: int = 12,
        num_domains: int = 2,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        # num_classes=0 → timm replaces the final fc with Identity and forward
        # returns globally-pooled features of dim `num_features` (512 here).
        self.encoder = timm.create_model(
            "resnet18", pretrained=pretrained, num_classes=0
        )
        feature_dim = getattr(self.encoder, "num_features", RESNET18_FEATURE_DIM)
        self.class_head = nn.Linear(feature_dim, num_classes)
        self.domain_head = DomainHead(feature_dim, num_domains=num_domains)

    def forward(self, x: torch.Tensor, lambda_: float = 0.0):
        features = self.encoder(x)
        class_logits = self.class_head(features)
        domain_logits = self.domain_head(features, lambda_)
        return class_logits, domain_logits, features
