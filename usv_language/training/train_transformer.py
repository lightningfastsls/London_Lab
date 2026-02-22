"""CLI training script for the autoregressive spectrogram transformer.

Usage:
    python -m usv_language.training.train_transformer \
        --data-dir /path/to/bout_spectrograms \
        --output-dir /path/to/checkpoints \
        --epochs 200

Loads pre-computed bout spectrograms + normalization stats from data-dir,
creates recording-based train/val splits, and trains the next-column
prediction transformer with masked MSE loss.

Supports single-GPU, DataParallel, and DDP training modes.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Bootstrap: ensure project root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from usv_language.data.dataset import (
    AugmentationConfig,
    BucketedBatchSampler,
    TransformerDataConfig,
    USVBoutDataset,
    split_recordings,
)
from usv_language.data.normalization import load_normalization_stats, normalize
from usv_language.models.transformer import SpectrogramTransformer, TransformerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cosine warmup scheduler (standalone copy, decoupled from v1 trainer)
# ---------------------------------------------------------------------------


class CosineWarmupScheduler:
    """Learning rate scheduler with linear warmup and cosine decay.

    During warmup (steps 1..warmup_steps), LR increases linearly from 0
    to base_lr. After warmup, LR decays following a cosine curve from
    base_lr down to min_lr.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.step_count = 0

    def step(self) -> None:
        """Advance one step and update the learning rate."""
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.base_lr * self.step_count / max(1, self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.max_steps - self.warmup_steps,
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def get_lr(self) -> float:
        """Return current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> dict:
        """Return scheduler state for checkpointing.

        Saves all schedule-defining fields so that resuming with different
        --epochs doesn't corrupt the cosine decay curve.
        """
        return {
            "step_count": self.step_count,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "min_lr": self.min_lr,
            "base_lr": self.base_lr,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore scheduler state from checkpoint."""
        self.step_count = state["step_count"]
        if "warmup_steps" in state:
            self.warmup_steps = state["warmup_steps"]
        if "max_steps" in state:
            self.max_steps = state["max_steps"]
        if "min_lr" in state:
            self.min_lr = state["min_lr"]
        if "base_lr" in state:
            self.base_lr = state["base_lr"]


# ---------------------------------------------------------------------------
# Masked MSE loss
# ---------------------------------------------------------------------------


def masked_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE loss only over non-padded frames.

    Parameters
    ----------
    predictions:
        Model output, shape (batch, seq_len, n_freq).
    targets:
        Ground truth next columns, shape (batch, seq_len, n_freq).
    mask:
        Float mask, shape (batch, seq_len). 1.0 = real, 0.0 = padding.

    Returns
    -------
    loss:
        Scalar mean MSE over valid frames.
    """
    # Expand mask to broadcast over frequency dimension
    mask_expanded = mask.unsqueeze(-1)  # (B, S, 1)
    sq_error = (predictions - targets) ** 2  # (B, S, n_freq)
    masked_error = sq_error * mask_expanded
    # Mean over all valid elements
    n_valid = mask_expanded.sum() * predictions.shape[-1]
    if n_valid == 0:
        return torch.tensor(0.0, device=predictions.device)
    return masked_error.sum() / n_valid


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_bout_spectrograms(
    data_dir: Path,
) -> tuple[list[np.ndarray], list[str]]:
    """Load pre-computed bout spectrograms from an HDF5 or directory.

    Expects either:
    - A directory of .npy files with naming: {recording_id}_bout{N}.npy
    - Or an HDF5 file (data_dir / "bout_spectrograms.h5")

    Each spectrogram has shape (n_freq, T).

    Returns (spectrograms, recording_ids).
    """
    import h5py

    h5_path = data_dir / "bout_spectrograms.h5"
    if h5_path.exists():
        spectrograms = []
        recording_ids = []
        with h5py.File(h5_path, "r") as f:
            for key in sorted(f.keys()):
                grp = f[key]
                spec = grp["spectrogram"][:]
                rec_id = grp.attrs.get("recording_id", key)
                spectrograms.append(spec.astype(np.float32))
                recording_ids.append(str(rec_id))
        return spectrograms, recording_ids

    # Fallback: directory of .npy files
    npy_files = sorted(data_dir.glob("*_bout*.npy"))
    if not npy_files:
        raise FileNotFoundError(
            f"No bout spectrograms found in {data_dir}. "
            f"Expected bout_spectrograms.h5 or *_bout*.npy files."
        )
    spectrograms = []
    recording_ids = []
    for p in npy_files:
        spec = np.load(str(p)).astype(np.float32)
        rec_id = p.stem.rsplit("_bout", 1)[0]
        spectrograms.append(spec)
        recording_ids.append(rec_id)
    return spectrograms, recording_ids


def build_dataloaders(
    spectrograms: list[np.ndarray],
    recording_ids: list[str],
    data_config: TransformerDataConfig,
    aug_config: AugmentationConfig,
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders with bucketed batching.

    Returns (train_loader, val_loader).
    """
    # Split recordings
    unique_ids = sorted(set(recording_ids))
    train_ids, val_ids, _ = split_recordings(unique_ids, data_config)
    train_id_set = set(train_ids)
    val_id_set = set(val_ids)

    train_specs = [s for s, r in zip(spectrograms, recording_ids) if r in train_id_set]
    train_recs = [r for r in recording_ids if r in train_id_set]
    val_specs = [s for s, r in zip(spectrograms, recording_ids) if r in val_id_set]
    val_recs = [r for r in recording_ids if r in val_id_set]

    train_dataset = USVBoutDataset(
        train_specs, train_recs, data_config,
        augmentation_config=aug_config, training=True, seed=data_config.seed,
    )
    val_dataset = USVBoutDataset(
        val_specs, val_recs, data_config,
        augmentation_config=None, training=False, seed=data_config.seed,
    )

    train_sampler = BucketedBatchSampler(
        train_dataset.get_lengths(), data_config.batch_size,
        shuffle=True, seed=data_config.seed,
    )
    val_sampler = BucketedBatchSampler(
        val_dataset.get_lengths(), data_config.batch_size,
        shuffle=False, seed=data_config.seed,
    )

    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler,
        num_workers=data_config.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_sampler=val_sampler,
        num_workers=data_config.num_workers, pin_memory=True,
    )

    logger.info(
        "Data: %d train chunks, %d val chunks, %d train recordings, %d val recordings",
        len(train_dataset), len(val_dataset), len(train_ids), len(val_ids),
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Optimizer setup
# ---------------------------------------------------------------------------


def build_optimizer(
    model: SpectrogramTransformer,
    lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Build AdamW optimizer with parameter groups.

    Weight matrices get weight_decay; biases, LayerNorm params,
    and embeddings get zero weight_decay.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No decay for biases, LayerNorm, and embeddings
        if "bias" in name or "norm" in name or "embed" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(param_groups, lr=lr)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineWarmupScheduler,
    device: torch.device,
    grad_clip: float,
) -> float:
    """Train for one epoch. Returns mean training loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        inputs = batch["input"].to(device)       # (B, S, n_freq)
        targets = batch["target"].to(device)      # (B, S, n_freq)
        mask = batch["mask"].to(device)           # (B, S)

        # Convert mask: dataset uses 1=real/0=pad, model expects True=pad
        padding_mask = ~mask.bool()

        optimizer.zero_grad()
        output, _ = model(inputs, attention_mask=padding_mask)

        loss = masked_mse_loss(output, targets, mask)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Validate model. Returns mean validation loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        mask = batch["mask"].to(device)

        padding_mask = ~mask.bool()
        output, _ = model(inputs, attention_mask=padding_mask)

        loss = masked_mse_loss(output, targets, mask)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineWarmupScheduler,
    epoch: int,
    config: TransformerConfig,
    best_val_loss: float,
    history: list[dict],
    patience_counter: int = 0,
) -> None:
    """Save a training checkpoint."""
    # Unwrap DataParallel if needed
    model_state = (
        model.module.state_dict()
        if hasattr(model, "module")
        else model.state_dict()
    )
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": config,
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
        "history": history,
    }
    torch.save(checkpoint, str(path))


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: CosineWarmupScheduler | None = None,
) -> dict:
    """Load a training checkpoint. Returns the checkpoint dict.

    Handles DataParallel-wrapped models: checkpoints are always saved
    without the ``module.`` prefix, so we unwrap before loading.
    """
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    # Unwrap DataParallel if needed (checkpoints save without module. prefix)
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    """Main training entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    # Configs
    model_config = TransformerConfig(
        n_freq=args.n_freq,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ffn=args.d_ffn,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    )
    data_config = TransformerDataConfig(
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    aug_config = AugmentationConfig(enabled=args.augment)

    # Load data
    data_dir = Path(args.data_dir)
    spectrograms, recording_ids = load_bout_spectrograms(data_dir)

    # Normalize if stats available
    stats_path = data_dir / "normalization_stats.npz"
    if stats_path.exists():
        stats = load_normalization_stats(stats_path)
        spectrograms = [normalize(s, stats) for s in spectrograms]
        logger.info("Applied per-frequency normalization from %s", stats_path)
    else:
        logger.warning("No normalization stats found at %s", stats_path)

    train_loader, val_loader = build_dataloaders(
        spectrograms, recording_ids, data_config, aug_config,
    )

    # Model
    model = SpectrogramTransformer(model_config).to(device)
    logger.info(
        "Model: %d parameters (%.1fM)",
        model.count_parameters(),
        model.count_parameters() / 1e6,
    )

    # Multi-GPU
    if args.data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info("Using DataParallel with %d GPUs", torch.cuda.device_count())

    # Optimizer and scheduler
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=args.warmup_steps,
        max_steps=total_steps, min_lr=args.min_lr,
    )

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    history: list[dict] = []
    patience_counter = 0

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            logger.warning(
                "Resume checkpoint not found: %s — starting from scratch",
                resume_path,
            )
        else:
            ckpt = load_checkpoint(resume_path, model, optimizer, scheduler)
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            patience_counter = ckpt.get("patience_counter", 0)
            history = ckpt.get("history", [])
            logger.info("Resumed from epoch %d", ckpt["epoch"])

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "model": {k: getattr(model_config, k) for k in model_config.__dataclass_fields__},
            "data": {k: getattr(data_config, k) for k in data_config.__dataclass_fields__},
            "training": {
                "lr": args.lr, "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps, "min_lr": args.min_lr,
                "grad_clip": args.grad_clip, "epochs": args.epochs,
                "patience": args.patience,
            },
        }, f, indent=2)

    # Training loop
    logger.info("Starting training for %d epochs", args.epochs)
    epoch = max(0, start_epoch - 1)  # default if loop doesn't run
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, args.grad_clip,
        )
        val_loss = validate(model, val_loader, device)
        lr = scheduler.get_lr()
        elapsed = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            "elapsed_s": elapsed,
        }
        history.append(record)

        logger.info(
            "Epoch %3d | train %.6f | val %.6f | lr %.2e | %.1fs",
            epoch, train_loss, val_loss, lr, elapsed,
        )

        # Checkpointing
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                output_dir / "best.pt", model, optimizer, scheduler,
                epoch, model_config, best_val_loss, history, patience_counter,
            )
        else:
            patience_counter += 1

        if (epoch + 1) % args.checkpoint_every == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch:04d}.pt", model, optimizer,
                scheduler, epoch, model_config, best_val_loss, history,
                patience_counter,
            )

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            logger.info(
                "Early stopping at epoch %d (patience=%d)",
                epoch, args.patience,
            )
            break

    # Save final checkpoint
    save_checkpoint(
        output_dir / "final.pt", model, optimizer, scheduler,
        epoch, model_config, best_val_loss, history, patience_counter,
    )

    # Save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(
        "Training complete. Best val loss: %.6f. Checkpoints in %s",
        best_val_loss, output_dir,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Train autoregressive spectrogram transformer",
    )

    # Data
    p.add_argument("--data-dir", type=str, required=True,
                    help="Directory with bout spectrograms + normalization stats")
    p.add_argument("--output-dir", type=str, required=True,
                    help="Directory for checkpoints and logs")

    # Model architecture
    p.add_argument("--n-freq", type=int, default=170)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--d-ffn", type=int, default=2048)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--augment", action="store_true", default=False)

    # Multi-GPU
    p.add_argument("--data-parallel", action="store_true", default=False)

    # Misc
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from")

    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
