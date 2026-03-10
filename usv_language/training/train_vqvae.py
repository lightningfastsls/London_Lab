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
import os
import re
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

_LAYER_FILENAME_RE = re.compile(r"hidden_states_layer(\d+)\.npy$")


def coerce_vqvae_config(raw_config: VQVAEConfig | dict | None) -> VQVAEConfig:
    """Normalize checkpoint config payloads across older/newer formats."""
    if raw_config is None:
        return VQVAEConfig()
    if isinstance(raw_config, VQVAEConfig):
        return raw_config
    if isinstance(raw_config, dict):
        return VQVAEConfig(**raw_config)
    raise TypeError(f"Unsupported VQ-VAE config payload: {type(raw_config)!r}")


def infer_hidden_state_layer(hidden_states_path: Path) -> int | None:
    """Infer transformer layer number from a hidden-state filename."""
    match = _LAYER_FILENAME_RE.search(hidden_states_path.name)
    if match is None:
        return None
    return int(match.group(1))


def build_hidden_state_provenance(hidden_states_path: Path) -> dict:
    """Capture provenance for the hidden-state artifact used in training."""
    hidden_states_path = hidden_states_path.resolve()
    provenance = {
        "hidden_states_path": str(hidden_states_path),
        "hidden_states_filename": hidden_states_path.name,
        "source_layer": infer_hidden_state_layer(hidden_states_path),
    }

    metadata_path = hidden_states_path.with_name("metadata.json")
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        provenance["metadata_path"] = str(metadata_path.resolve())
        provenance["metadata_d_model"] = metadata.get("d_model")
        provenance["metadata_total_frames"] = metadata.get("total_frames")
        provenance["layers_extracted"] = metadata.get("layers_extracted")

    return provenance


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
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        self.path = path
        self.window_size = window_size
        self.stride = stride

        # Memory-map the file (read-only)
        self._mmap = np.load(path, mmap_mode="r")
        if self._mmap.ndim != 2:
            raise ValueError(
                f"Hidden states array must have shape (frames, d_model), got {self._mmap.shape}"
            )
        self.total_frames, self.d_model = self._mmap.shape
        if self.total_frames == 0:
            raise ValueError("Hidden states array is empty")
        if self.d_model == 0:
            raise ValueError("Hidden states array must have non-zero feature width")

        # Compute window start indices
        if self.total_frames < window_size:
            # Short file: single padded window
            self.starts = [0]
        else:
            self.starts = list(range(0, self.total_frames - window_size + 1, stride))

            # Ensure tail coverage: append a final window anchored to the end
            # when the stride grid would otherwise drop the remainder.
            final_start = self.total_frames - window_size
            if self.starts[-1] != final_start:
                self.starts.append(final_start)

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

    Uses the last ``val_fraction`` windows for validation and drops a gap of
    overlapping windows at the train/val boundary so that no validation window
    shares frames with any training window.

    Returns (train_loader, val_loader).
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    n = len(dataset)
    if n < 2:
        raise ValueError(
            "HiddenStateDataset needs at least 2 windows for train/val split; "
            f"got {n}"
        )

    n_val = max(1, int(n * val_fraction))
    first_val_idx = max(1, n - n_val)

    # Consecutive windows overlap whenever stride < window_size. Walk the
    # boundary back until the last train window ends before the first val
    # window starts. This handles the final end-anchored window as well as
    # regular stride-aligned windows.
    first_val_start = dataset.starts[first_val_idx]
    train_end_exclusive = first_val_idx
    while (
        train_end_exclusive > 0
        and dataset.starts[train_end_exclusive - 1] + dataset.window_size > first_val_start
    ):
        train_end_exclusive -= 1

    gap_windows = first_val_idx - train_end_exclusive

    if train_end_exclusive == 0:
        raise ValueError(
            "Not enough windows to create independent train/val splits after "
            f"dropping {gap_windows} overlapping boundary windows; got {n} windows"
        )

    train_dataset = torch.utils.data.Subset(dataset, list(range(train_end_exclusive)))
    val_dataset = torch.utils.data.Subset(dataset, list(range(first_val_idx, n)))

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
        "Data: %d train windows, %d val windows, %d gap windows dropped (total %d)",
        len(train_dataset), len(val_dataset), first_val_idx - train_end_exclusive, n,
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
    provenance: dict | None = None,
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
    if provenance is not None:
        checkpoint["provenance"] = provenance
    torch.save(checkpoint, str(path))


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: CosineWarmupScheduler | None = None,
) -> dict:
    """Load a training checkpoint. Returns the checkpoint dict."""
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    checkpoint["config"] = coerce_vqvae_config(checkpoint.get("config"))
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

    hidden_states_path = Path(args.hidden_states)
    if not hidden_states_path.exists():
        raise FileNotFoundError(f"Hidden states file not found: {hidden_states_path}")
    provenance = build_hidden_state_provenance(hidden_states_path)

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
        str(hidden_states_path),
        window_size=args.window_size,
        stride=args.stride,
    )
    if dataset.d_model != model_config.d_model:
        raise ValueError(
            "Hidden-state width does not match requested d_model: "
            f"{dataset.d_model} vs {model_config.d_model}"
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
            checkpoint_config = ckpt["config"]
            if checkpoint_config != model_config:
                raise ValueError(
                    "Resume checkpoint model config does not match requested "
                    "training config. Resume with the original architecture or "
                    "start a new run."
                )
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            patience_counter = ckpt.get("patience_counter", 0)
            history = ckpt.get("history", [])
            logger.info("Resumed from epoch %d", ckpt["epoch"])
            if start_epoch >= args.epochs:
                raise ValueError(
                    f"Checkpoint already reached epoch {ckpt['epoch']}. "
                    f"Increase --epochs above {ckpt['epoch'] + 1} to resume."
                )
            checkpoint_provenance = ckpt.get("provenance")
            if checkpoint_provenance is not None:
                ckpt_hs_path = checkpoint_provenance.get("hidden_states_path")
                cur_hs_path = provenance.get("hidden_states_path")
                if ckpt_hs_path and cur_hs_path:
                    try:
                        paths_match = os.path.samefile(ckpt_hs_path, cur_hs_path)
                    except (FileNotFoundError, OSError):
                        paths_match = Path(ckpt_hs_path).name == Path(cur_hs_path).name
                    if not paths_match:
                        raise ValueError(
                            "Resume checkpoint hidden-state provenance does not match "
                            "the requested hidden-state artifact. "
                            f"Checkpoint: {ckpt_hs_path}, current: {cur_hs_path}. "
                            "Resume with the original hidden_states file or start a new run."
                        )
                ckpt_layer = checkpoint_provenance.get("source_layer")
                cur_layer = provenance.get("source_layer")
                if ckpt_layer is not None and cur_layer is not None and ckpt_layer != cur_layer:
                    raise ValueError(
                        "Resume checkpoint source_layer does not match the current "
                        f"hidden-state artifact: checkpoint={ckpt_layer}, current={cur_layer}. "
                        "Resume with the original hidden_states file or start a new run."
                    )

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
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
            "provenance": provenance,
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
                epoch, model_config, best_val_loss, history, provenance, patience_counter,
            )
        else:
            patience_counter += 1

        if (epoch + 1) % args.checkpoint_every == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch:04d}.pt", model, optimizer,
                scheduler, epoch, model_config, best_val_loss, history,
                provenance,
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
        epoch, model_config, best_val_loss, history, provenance, patience_counter,
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

