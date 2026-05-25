"""Module 18.3 — ResNet-18 model factory for the 12-class lab USV classifier.

ROADMAP §18.3 file 1. The factory wraps timm's ImageNet-pretrained ResNet-18
backbone and swaps the classification head for the Grimsley 2011 12-class
syllable taxonomy. Used by:
  - scripts/train_lab_classifier.py (training entry point)
  - src/usv_spectrogram/classifier/training.py (train_classifier loop)

Reference: He et al. 2015 (arXiv:1512.03385), timm.create_model docs.
"""

from __future__ import annotations

import timm
import torch.nn as nn

NUM_CLASSES = 12


def build_resnet18_classifier(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """Return a timm ResNet-18 with a `num_classes`-way classification head.

    Parameters
    ----------
    num_classes
        Output dim of the final fully-connected layer. Default 12 = Grimsley
        2011 syllable taxonomy.
    pretrained
        If True, load ImageNet-pretrained backbone weights via timm. Set to
        False for CI / offline / smoke-test environments.

    Returns
    -------
    nn.Module
        Model whose ``forward(x)`` accepts a ``(B, 3, H, W)`` tensor and
        returns logits of shape ``(B, num_classes)``.
    """
    return timm.create_model(
        "resnet18",
        pretrained=pretrained,
        num_classes=num_classes,
    )
