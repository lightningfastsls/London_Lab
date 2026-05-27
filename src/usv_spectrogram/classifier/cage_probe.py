"""Module 18.4 — Linear cage probe on frozen encoder features.

ROADMAP §18.4. The cage probe is the *feature-level* cage-invariance gate for
DANN (the pixel-level gate is the 18.1 cleaning validation; the residual that
DANN must remove lives in the encoder's learned features).

Idea: freeze the trained encoder, then train a *linear* classifier on its
features to predict the recording cage. If the encoder is cage-invariant, even
a linear probe cannot recover the cage → accuracy near chance (~0.50 for two
balanced cages). The 18.4 pass threshold is **< 0.65**; lower is better.

Why linear (not a deep probe): a linear probe measures whether cage identity is
*linearly decodable* from the representation. A powerful non-linear probe could
recover cage from almost any representation, so it would not distinguish a
cage-invariant encoder from a cage-entangled one. Linear decodability is the
standard operationalisation (Alain & Bengio 2016, "Understanding intermediate
layers using linear classifier probes").

Implementation note — the encoder must stay *frozen*. We extract all features
once under ``torch.no_grad()`` into cached tensors and train the linear head on
those, so the encoder never participates in an optimizer step. This is both
correct (the measurement is of fixed features) and cheap.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Linear-probe training recipe. Weight decay matters: on a signal-free
# representation it pulls the probe toward the chance/majority solution
# (so the negative control lands near 0.50), while a genuine cage signal still
# overcomes it (positive control reaches >0.90).
_PROBE_EPOCHS = 200
_PROBE_LR = 1e-2
_PROBE_WEIGHT_DECAY = 1e-2
_STD_EPS = 1e-6


def _extract_features(
    encoder: nn.Module,
    loader,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the frozen encoder over ``loader`` and return ``(features, labels)``.

    Features are detached and moved to ``device``; the encoder is run in eval
    mode under ``no_grad`` so it neither updates BatchNorm running stats nor
    enters the autograd graph.
    """
    encoder.eval()
    feats, labels = [], []
    with torch.no_grad():
        for batch in loader:
            images, cage = batch[0], batch[1]
            images = images.to(device)
            out = encoder(images)
            feats.append(out.reshape(out.shape[0], -1).detach())
            labels.append(cage.to(device))
    if not feats:
        empty_dim = 0
        return (
            torch.empty(0, empty_dim, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def linear_cage_probe(
    encoder: nn.Module,
    train_loader,
    val_loader,
    num_cages: int = 2,
    device: str = "cpu",
) -> float:
    """Train a linear probe on frozen encoder features to predict cage.

    Parameters
    ----------
    encoder
        A feature extractor mapping ``(B, C, H, W) -> (B, D)`` (or any shape
        that flattens to ``(B, D)``). Frozen — its weights are never updated.
    train_loader, val_loader
        Yield ``(images, cage_label)`` batches.
    num_cages
        Number of cage classes (D4 default: 2).
    device
        Torch device string. Defaults to ``"cpu"`` (this pipeline's CPU box has
        no GPU); pass ``"cuda"`` on the rig.

    Returns
    -------
    float
        Validation accuracy in ``[0, 1]``. Pass threshold for 18.4: ``< 0.65``.
        Lower ⇒ more cage-invariant encoder. Returns ``0.0`` if the validation
        set is empty.
    """
    encoder = encoder.to(device)

    train_feats, train_labels = _extract_features(encoder, train_loader, device)
    val_feats, val_labels = _extract_features(encoder, val_loader, device)

    # Empty validation set → no measurement possible.
    if val_feats.shape[0] == 0:
        return 0.0

    feature_dim = train_feats.shape[1] if train_feats.shape[0] > 0 else val_feats.shape[1]

    # Standardise with train statistics so the probe optimisation is well-scaled
    # regardless of the encoder's output magnitude.
    if train_feats.shape[0] > 0:
        mean = train_feats.mean(dim=0, keepdim=True)
        std = train_feats.std(dim=0, keepdim=True).clamp_min(_STD_EPS)
    else:
        mean = torch.zeros(1, feature_dim, device=device)
        std = torch.ones(1, feature_dim, device=device)

    train_x = (train_feats - mean) / std
    val_x = (val_feats - mean) / std

    # Linear probe head — the ONLY module that gets optimised.
    probe = nn.Linear(feature_dim, num_cages).to(device)
    optimizer = torch.optim.Adam(
        probe.parameters(), lr=_PROBE_LR, weight_decay=_PROBE_WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()

    if train_feats.shape[0] > 0:
        probe.train()
        for _ in range(_PROBE_EPOCHS):
            optimizer.zero_grad()
            logits = probe(train_x)
            loss = criterion(logits, train_labels)
            loss.backward()
            optimizer.step()

    # Validation accuracy.
    probe.eval()
    with torch.no_grad():
        val_logits = probe(val_x)
        preds = val_logits.argmax(dim=1)
        correct = (preds == val_labels).sum().item()
    accuracy = correct / val_labels.shape[0]
    return float(accuracy)
