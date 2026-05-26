"""Module 18.3 — class-weighted focal loss for the lab classifier.

ROADMAP §18.3 file 3. Implements Lin et al. 2017 focal loss with per-class
weighting (D5 strategy from PLAN §"Minority class handling"):

    focal_loss = mean_b( w[t_b] * (1 - p_{t_b})^gamma * (-log p_{t_b}) )

Notes on reduction
------------------
Reduction is plain-mean over the batch (``sum / B``), NOT PyTorch's default
weighted-mean (``sum / sum(weights[targets])``). Plain-mean is required so
that class weights amplify gradient magnitudes — under PyTorch's weighted-
mean reduction the per-sample weight ``w[t]`` cancels in the single-sample
limit (numerator and denominator share ``w[t]``), defeating the D5 strategy.
See tests/classifier/test_losses.py::test_focal_loss_gamma0_equals_weighted_ce
for the matching reference reduction.

Reference: Lin et al. 2017 (arXiv:1708.02002).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Class-weighted focal loss, plain-mean over the batch.

    Parameters
    ----------
    logits
        Unnormalised classifier outputs of shape ``(B, C)``.
    targets
        Integer class labels of shape ``(B,)`` with values in ``[0, C)``.
    class_weights
        Per-class scaling factors of shape ``(C,)``. Larger values up-weight
        the corresponding class.
    gamma
        Focusing parameter. ``gamma=0`` reduces to plain-mean weighted CE.
        ``gamma=2.0`` is the Lin et al. 2017 default.

    Returns
    -------
    torch.Tensor
        A scalar (0-dim) tensor — the mean focal loss over the batch.

    Raises
    ------
    ValueError
        If ``logits.shape[0] != targets.shape[0]`` or
        ``class_weights.shape[0] != logits.shape[1]``.
    """
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            f"logits.shape[0] ({logits.shape[0]}) must equal "
            f"targets.shape[0] ({targets.shape[0]})"
        )
    if class_weights.shape[0] != logits.shape[1]:
        raise ValueError(
            f"class_weights.shape[0] ({class_weights.shape[0]}) must equal "
            f"logits.shape[1] ({logits.shape[1]})"
        )

    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    p_t = torch.exp(-nll)
    sample_w = class_weights[targets]
    focal_term = (1.0 - p_t).clamp(min=0.0) ** gamma
    per_sample = sample_w * focal_term * nll
    return per_sample.mean()
