"""CLI training script for the hidden-state VQ-VAE.

Usage:
    python -m usv_language.training.train_vqvae \
        --hidden-states /path/to/hidden_states_layer6.npy \
        --output-dir /path/to/vqvae_checkpoints \
        --epochs 100

Loads pre-extracted transformer hidden states (shape: total_frames x 512),
chunks them into overlapping windows, and trains a VQ-VAE to discover
discrete "concepts" in the transformer's internal representations.

The transformer is frozen — this is a post-hoc interpretability tool.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Bootstrap: ensure project root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from usv_language.models.vqvae import HiddenStateVQVAE, VQVAEConfig
from usv_language.training.train_transformer import CosineWarmupScheduler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class HiddenStateDataset(Dataset):
    """Dataset of windowed hidden-state chunks from a memory-mapped .npy file.

    Loads a pre-extracted .npy file of shape (total_frames, d_model) and
    chunks it into overlapping windows for VQ-VAE training.

    Parameters
    ----------
    path:
        Path to .npy file with shape (total_frames, d_model).
    window_size:
        Number of frames per training sample.
    stride:
        Step size between consecutive windows (< window_size = overlap).
    """

    def __init__(
        self,
        path: str,
        window_size: int = 128,
        stride: int = 64,
    ) -> None:
        self.path = path
        self.window_size = window_size
        self.stride = stride

        # Memory-map the file (read-only)
        self._mmap = np.load(path, mmap_mode="r")
        self.total_frames, self.d_model = self._mmap.shape

        # Compute window start indices
        if self.total_frames < window_size:
            # Short file: single padded window
            self.starts = [0]
        else:
            self.starts = list(range(0, self.total_frames - window_size + 1, stride))

    def close(self) -> None:
        """Release the memory-mapped file handle (important on Windows)."""
        if hasattr(self, "_mmap") and self._mmap is not None:
            # numpy mmap is backed by a memmap; delete to release file handle
            del self._mmap
            self._mmap = None

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = self.starts[idx]
        end = start + self.window_size

        if end <= self.total_frames:
            # Normal window: .copy() avoids Windows mmap file handle issues
            chunk = self._mmap[start:end].copy()
        else:
            # Padded window (short file)
            available = self.total_frames - start
            chunk = np.zeros((self.window_size, self.d_model), dtype=np.float32)
            chunk[:available] = self._mmap[start : start + available].copy()

        return torch.from_numpy(chunk.astype(np.float32))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def build_dataloaders(
    dataset: HiddenStateDataset,
    batch_size: int,
    val_fraction: float = 0.1,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Split dataset sequentially and build DataLoaders.

    First (1 - val_fraction) for training, last val_fraction for validation.
    Sequential split avoids temporal leakage.

    Returns (train_loader, val_loader).
    """
    n = len(dataset)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val

    train_dataset = torch.utils.data.Subset(dataset, list(range(n_train)))
    val_dataset = torch.utils.data.Subset(dataset, list(range(n_train, n)))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    logger.info(
        "Data: %d train windows, %d val windows (total %d)",
        n_train, n_val, n,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# K-means initialization
# ---------------------------------------------------------------------------


def collect_init_samples(
    dataset: HiddenStateDataset,
    n_samples: int = 5000,
) -> torch.Tensor:
    """Gather samples for k-means codebook initialization.

    Randomly samples individual frames from the dataset to get a
    representative set of hidden-state vectors.

    Returns tensor of shape (n_samples, d_model).
    """
    all_frames = []
    # Collect frames from random windows
    indices = np.random.choice(len(dataset), size=min(len(dataset), 100), replace=False)
    for idx in indices:
        window = dataset[idx]  # (window_size, d_model)
        all_frames.append(window)
        if sum(f.shape[0] for f in all_frames) >= n_samples:
            break

    stacked = torch.cat(all_frames, dim=0)
    if stacked.shape[0] > n_samples:
        perm = torch.randperm(stacked.shape[0])[:n_samples]
        stacked = stacked[perm]

    return stacked


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def build_optimizer(
    model: HiddenStateVQVAE,
    lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Build AdamW optimizer, excluding codebook (EMA-updated, not gradient).

    The quantizer.embedding.weight is updated via EMA in the forward pass,
    so it must NOT be included in the optimizer's parameter groups.
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Skip codebook — it's EMA-updated
        if "quantizer.embedding" in name:
            continue
        if "bias" in name or "norm" in name:
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
) -> dict:
    """Train for one epoch. Returns dict of mean metrics."""
    model.train()
    totals = {"recon_loss": 0.0, "commit_loss": 0.0, "total_loss": 0.0,
              "perplexity": 0.0, "codebook_usage": 0.0}
    n_batches = 0

    for batch in loader:
        x = batch.to(device)  # (B, window_size, d_model)

        optimizer.zero_grad()
        out = model(x)
        out["total_loss"].backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        for key in totals:
            totals[key] += out[key].item()
        n_batches += 1

    return {k: v / max(1, n_batches) for k, v in totals.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Validate model. Returns dict of mean metrics."""
    model.eval()
    totals = {"recon_loss": 0.0, "commit_loss": 0.0, "total_loss": 0.0,
              "perplexity": 0.0, "codebook_usage": 0.0}
    n_batches = 0

    for batch in loader:
        x = batch.to(device)
        out = model(x)
        for key in totals:
            totals[key] += out[key].item()
        n_batches += 1

    return {k: v / max(1, n_batches) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineWarmupScheduler,
    epoch: int,
    config: VQVAEConfig,
    best_val_loss: float,
    history: list[dict],
    patience_counter: int = 0,
) -> None:
    """Save a training checkpoint."""
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
    """Load a training checkpoint. Returns the checkpoint dict."""
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
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


def train(args: argparse.Namespace) -> dict:
    """Main training entry point. Returns final validation metrics dict.

    The return value is used by compare_layers.py to collect metrics
    across multiple layer trainings.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    # Config
    model_config = VQVAEConfig(
        d_model=args.d_model,
        codebook_size=args.codebook_size,
        codebook_dim=args.codebook_dim,
        commitment_weight=args.commitment_weight,
        ema_decay=args.ema_decay,
        dead_code_threshold=args.dead_code_threshold,
        use_conv_encoder=args.use_conv_encoder,
        conv_kernel_size=args.conv_kernel_size,
    )

    # Dataset
    dataset = HiddenStateDataset(
        args.hidden_states,
        window_size=args.window_size,
        stride=args.stride,
    )
    train_loader, val_loader = build_dataloaders(
        dataset,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
    )

    # Model
    model = HiddenStateVQVAE(model_config).to(device)
    logger.info(
        "Model: %d parameters (%.1fK)",
        model.count_parameters(),
        model.count_parameters() / 1e3,
    )

    # K-means initialization
    logger.info("Collecting samples for k-means initialization...")
    init_samples = collect_init_samples(dataset, n_samples=args.init_samples)
    model.initialize_codebook_from_data(init_samples.to(device))
    logger.info("Codebook initialized via k-means")

    # Optimizer (excludes codebook — EMA-updated)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=args.warmup_steps,
        max_steps=total_steps, min_lr=args.min_lr,
    )

    # Resume
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
            "model": {k: getattr(model_config, k)
                      for k in model_config.__dataclass_fields__},
            "training": {
                "lr": args.lr, "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps, "min_lr": args.min_lr,
                "grad_clip": args.grad_clip, "epochs": args.epochs,
                "patience": args.patience,
            },
            "dataset": {
                "hidden_states": args.hidden_states,
                "window_size": args.window_size,
                "stride": args.stride,
            },
        }, f, indent=2)

    # Training loop
    logger.info("Starting training for %d epochs", args.epochs)
    epoch = max(0, start_epoch - 1)
    final_metrics = {}

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, args.grad_clip,
        )
        val_metrics = validate(model, val_loader, device)
        lr = scheduler.get_lr()
        elapsed = time.time() - t0

        record = {
            "epoch": epoch,
            "lr": lr,
            "elapsed_s": elapsed,
        }
        for k, v in train_metrics.items():
            record[f"train_{k}"] = v
        for k, v in val_metrics.items():
            record[f"val_{k}"] = v
        history.append(record)

        logger.info(
            "Epoch %3d | recon %.4f / %.4f | perp %.1f | usage %.1f%% | lr %.2e | %.1fs",
            epoch,
            train_metrics["recon_loss"],
            val_metrics["recon_loss"],
            val_metrics["perplexity"],
            val_metrics["codebook_usage"] * 100,
            lr,
            elapsed,
        )

        # Checkpointing
        is_best = val_metrics["total_loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["total_loss"]
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

        final_metrics = val_metrics

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            logger.info(
                "Early stopping at epoch %d (patience=%d)",
                epoch, args.patience,
            )
            break

    # Save final
    save_checkpoint(
        output_dir / "final.pt", model, optimizer, scheduler,
        epoch, model_config, best_val_loss, history, patience_counter,
    )

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(
        "Training complete. Best val loss: %.6f. Checkpoints in %s",
        best_val_loss, output_dir,
    )

    return final_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Train hidden-state VQ-VAE for transformer interpretability",
    )

    # Data
    p.add_argument("--hidden-states", type=str, required=True,
                   help="Path to .npy file with hidden states (total_frames, 512)")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Directory for checkpoints and logs")

    # Model
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--codebook-size", type=int, default=64)
    p.add_argument("--codebook-dim", type=int, default=64)
    p.add_argument("--commitment-weight", type=float, default=0.25)
    p.add_argument("--ema-decay", type=float, default=0.99)
    p.add_argument("--dead-code-threshold", type=float, default=2.0)
    p.add_argument("--use-conv-encoder", action="store_true", default=True)
    p.add_argument("--no-conv-encoder", dest="use_conv_encoder",
                   action="store_false")
    p.add_argument("--conv-kernel-size", type=int, default=5)

    # Dataset
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--init-samples", type=int, default=5000,
                   help="Samples for k-means codebook init")

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--val-fraction", type=float, default=0.1)

    # Misc
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (0 recommended on Windows)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")

    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
