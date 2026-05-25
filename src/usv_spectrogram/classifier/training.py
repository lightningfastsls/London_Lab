"""Module 18.3 — training loop for the lab USV syllable classifier.

ROADMAP §18.3 file 4. Wires the model factory, focal loss, and augmentation
modules into an AdamW + cosine-LR + warmup loop with early stopping on macro
F1. Held-out 845 evaluation is label-distribution based for the smoke path;
Stream V (GPU real-data) will extend it to load patches and run inference.

Reference: PLAN §"Phase 1.2 — Baseline ResNet-18 Training".
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from .losses import focal_loss
from .model import NUM_CLASSES, build_resnet18_classifier


@dataclass(frozen=True)
class TrainingConfig:
    """Frozen hyperparameter bundle for the training loop.

    All fields default to PLAN §"Phase 1.2" recommendations. ``device='auto'``
    is supported as a runtime indicator that resolves to 'cuda' when available
    and 'cpu' otherwise.
    """

    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 0
    cosine_min_lr: float = 1e-5
    early_stop_patience: int = 8
    confusion_matrix_every_epochs: int = 5
    focal_gamma: float = 2.0
    device: str = "cuda"
    pretrained: bool = True

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.warmup_epochs < 0:
            raise ValueError(
                f"warmup_epochs must be >= 0, got {self.warmup_epochs}"
            )
        if self.warmup_epochs > self.epochs:
            raise ValueError(
                f"warmup_epochs ({self.warmup_epochs}) must be <= epochs "
                f"({self.epochs}); otherwise cosine schedule never activates."
            )
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError(
                f"device must be one of {{'auto', 'cpu', 'cuda'}}, got {self.device!r}"
            )


_GRIMSLEY_12_DISPLAY = (
    "Noise", "Step up", "Down-FM", "Short", "Chevron", "Up-FM",
    "Flat", "Two steps", "Step down", "Complex", "Reverse Chevron",
    "Multi-steps",
)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _compute_class_weights(loader: DataLoader, n_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights normalised so the sum equals n_classes.

    Iterates the underlying dataset once. Falls back to uniform if labels
    cannot be inferred (e.g., dataset is an IterableDataset).
    """
    counts = np.zeros(n_classes, dtype=np.float64)
    try:
        dataset = loader.dataset
        if hasattr(dataset, "tensors") and len(dataset.tensors) >= 2:
            labels = dataset.tensors[1].cpu().numpy().astype(int)
        else:
            labels = np.array([int(dataset[i][1]) for i in range(len(dataset))])
        for c in labels:
            if 0 <= c < n_classes:
                counts[c] += 1
    except Exception:
        return torch.ones(n_classes, dtype=torch.float32)

    counts = np.clip(counts, 1.0, None)
    inv = counts.sum() / (n_classes * counts)
    inv = inv * (n_classes / inv.sum())
    return torch.as_tensor(inv, dtype=torch.float32)


def _cosine_warmup_lr(epoch: int, cfg: TrainingConfig) -> float:
    """LR multiplier with linear warmup then cosine decay to ``cosine_min_lr``."""
    if cfg.warmup_epochs > 0 and epoch < cfg.warmup_epochs:
        return (epoch + 1) / max(1, cfg.warmup_epochs)
    progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_ratio = cfg.cosine_min_lr / cfg.learning_rate
    return min_ratio + (1.0 - min_ratio) * cosine


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    class_weights: torch.Tensor,
    device: torch.device,
    focal_gamma: float,
) -> float:
    model.train()
    total = 0.0
    n_batches = 0
    for batch in loader:
        images, targets = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = focal_loss(logits, targets, class_weights.to(device), gamma=focal_gamma)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().item())
        n_batches += 1
    return total / max(1, n_batches)


@torch.no_grad()
def _collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    for batch in loader:
        images, targets = batch[0].to(device), batch[1].to(device)
        logits = model(images)
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        trues.append(targets.cpu().numpy())
    if not preds:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(preds), np.concatenate(trues)


def _split_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    if y_true.size == 0:
        return {
            "macro_f1": 0.0,
            "per_class_precision": [0.0] * NUM_CLASSES,
            "per_class_recall": [0.0] * NUM_CLASSES,
            "confusion_matrix": np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int).tolist(),
        }
    labels = list(range(NUM_CLASSES))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0,
    )
    cm = sk_confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "macro_f1": float(np.mean(f1)),
        "per_class_precision": [float(x) for x in precision],
        "per_class_recall": [float(x) for x in recall],
        "confusion_matrix": cm.tolist(),
    }


def _evaluate_held_out_845(csv_path: Path) -> dict[str, float]:
    """Label-distribution stats from the held-out CSV.

    Used by the smoke path where no audio patches are available. Stream V
    will extend this to load patches and run real inference.
    """
    df = pd.read_csv(csv_path)
    label_col = "call_label" if "call_label" in df.columns else df.columns[0]
    verdict_col = "usv_verdict" if "usv_verdict" in df.columns else None

    counts = df[label_col].value_counts(normalize=True).values.astype(np.float64)
    counts = counts[counts > 0]
    entropy = float(-np.sum(counts * np.log(counts)))
    entropy = min(entropy, math.log(max(2, NUM_CLASSES)))

    if verdict_col is not None:
        predicted_usv = (df[label_col].astype(str).str.lower() != "noise").astype(int)
        actual_usv = (df[verdict_col].astype(str).str.lower() == "usv").astype(int)
        usv_noise_acc = float((predicted_usv == actual_usv).mean()) if len(df) else 0.0
    else:
        usv_noise_acc = 0.0

    return {"usv_noise_acc": usv_noise_acc, "syllable_entropy_mean": entropy}


def train_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainingConfig,
    output_dir: Path,
    held_out_845_csv: Path,
) -> dict[str, Any]:
    """Train the lab classifier and return a metrics dict + write best.pt.

    Returns
    -------
    dict
        Contains: macro_f1_val, macro_f1_test, per_class_precision,
        per_class_recall, confusion_matrix, usv_noise_acc,
        syllable_entropy_mean.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(cfg.device)

    model = build_resnet18_classifier(
        num_classes=NUM_CLASSES, pretrained=cfg.pretrained
    ).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda e: _cosine_warmup_lr(e, cfg),
    )
    class_weights = _compute_class_weights(train_loader, NUM_CLASSES)

    best_f1 = -math.inf
    epochs_no_improve = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(cfg.epochs):
        train_loss = _train_one_epoch(
            model, train_loader, optimizer, class_weights, device, cfg.focal_gamma,
        )
        scheduler.step()

        val_pred, val_true = _collect_predictions(model, val_loader, device)
        val_metrics = _split_metrics(val_true, val_pred)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "macro_f1_val": val_metrics["macro_f1"]}
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stop_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "history": history}, output_dir / "best.pt")

    val_pred, val_true = _collect_predictions(model, val_loader, device)
    val_metrics = _split_metrics(val_true, val_pred)
    test_pred, test_true = _collect_predictions(model, test_loader, device)
    test_metrics = _split_metrics(test_true, test_pred)
    held_out = _evaluate_held_out_845(Path(held_out_845_csv))

    result = {
        "macro_f1_val": float(val_metrics["macro_f1"]),
        "macro_f1_test": float(test_metrics["macro_f1"]),
        "per_class_precision": test_metrics["per_class_precision"],
        "per_class_recall": test_metrics["per_class_recall"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "usv_noise_acc": float(held_out["usv_noise_acc"]),
        "syllable_entropy_mean": float(held_out["syllable_entropy_mean"]),
        "history": history,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump({k: v for k, v in result.items() if k != "history"}, f, indent=2)
    return result
