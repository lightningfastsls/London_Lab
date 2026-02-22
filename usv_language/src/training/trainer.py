"""Training loop for VQ-VAE with staged progression.

Includes AdamW optimizer, cosine warmup scheduler, early stopping,
gradient clipping, checkpoint save/load, and metric logging.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from usv_language.src.model.vqvae import VQVAE
from usv_language.src.training.losses import VQVAELoss

logger = logging.getLogger(__name__)


class CosineWarmupScheduler:
    """Learning rate scheduler with linear warmup and cosine decay."""

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
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.base_lr * self.step_count / max(1, self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.max_steps - self.warmup_steps
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr


class VQVAETrainer:
    """Training loop for VQ-VAE.

    Parameters
    ----------
    model:
        VQVAE model instance.
    train_loader:
        Training DataLoader.
    val_loader:
        Validation DataLoader (optional).
    lr:
        Learning rate.
    weight_decay:
        AdamW weight decay.
    gradient_clip_norm:
        Max gradient norm for clipping.
    warmup_steps:
        Number of warmup steps for LR scheduler.
    min_lr:
        Minimum learning rate after cosine decay.
    loss_weights:
        Dict with keys: reconstruction, vq_commitment, next_code_prediction.
    device:
        Training device ("cpu" or "cuda").
    checkpoint_dir:
        Directory for saving checkpoints.
    early_stopping_patience:
        Stop training after this many epochs without improvement.
    """

    def __init__(
        self,
        model: VQVAE,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        gradient_clip_norm: float = 1.0,
        warmup_steps: int = 500,
        min_lr: float = 1e-6,
        loss_weights: Optional[Dict[str, float]] = None,
        device: str = "cpu",
        checkpoint_dir: Optional[str] = None,
        early_stopping_patience: int = 10,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gradient_clip_norm = gradient_clip_norm
        self.early_stopping_patience = early_stopping_patience

        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None

        # Loss
        lw = loss_weights or {}
        self.criterion = VQVAELoss(
            recon_weight=lw.get("reconstruction", 1.0),
            vq_weight=lw.get("vq_commitment", 1.0),
            next_code_weight=lw.get("next_code_prediction", 0.1),
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )

        # Scheduler
        max_steps = len(train_loader) * 100  # Estimate
        self.scheduler = CosineWarmupScheduler(
            self.optimizer, warmup_steps, max_steps, min_lr,
        )

        # History
        self.history: List[Dict[str, float]] = []

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        totals = {"total_loss": 0, "recon_loss": 0, "vq_loss": 0,
                  "next_code_loss": 0, "perplexity": 0, "codebook_usage": 0}
        n_batches = 0

        for batch in self.train_loader:
            x = batch["spectrogram"].to(self.device)
            out = self.model(x)
            losses = self.criterion(out, x)

            self.optimizer.zero_grad()
            losses["total_loss"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
            self.optimizer.step()
            self.scheduler.step()

            totals["total_loss"] += losses["total_loss"].item()
            totals["recon_loss"] += losses["recon_loss"].item()
            totals["vq_loss"] += losses["vq_loss"].item()
            totals["next_code_loss"] += losses["next_code_loss"].item()
            totals["perplexity"] += out["perplexity"].item()
            totals["codebook_usage"] += out["codebook_usage"].item()
            n_batches += 1

        return {k: v / max(1, n_batches) for k, v in totals.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        totals = {"total_loss": 0, "recon_loss": 0, "vq_loss": 0,
                  "next_code_loss": 0, "perplexity": 0, "codebook_usage": 0}
        n_batches = 0

        for batch in self.val_loader:
            x = batch["spectrogram"].to(self.device)
            out = self.model(x)
            losses = self.criterion(out, x)

            totals["total_loss"] += losses["total_loss"].item()
            totals["recon_loss"] += losses["recon_loss"].item()
            totals["vq_loss"] += losses["vq_loss"].item()
            totals["next_code_loss"] += losses["next_code_loss"].item()
            totals["perplexity"] += out["perplexity"].item()
            totals["codebook_usage"] += out["codebook_usage"].item()
            n_batches += 1

        return {f"val_{k}": v / max(1, n_batches) for k, v in totals.items()}

    def save_checkpoint(self, epoch: int, path: Path) -> None:
        """Save model checkpoint."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_step_count": self.scheduler.step_count,
            "history": self.history,
            "config": self.model.config,
        }, path)

    def load_checkpoint(self, path: str | Path) -> int:
        """Load checkpoint. Returns the epoch number."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.step_count = checkpoint["scheduler_step_count"]
        self.history = checkpoint["history"]
        return checkpoint["epoch"]

    def train(
        self,
        max_epochs: int = 100,
        checkpoint_every: int = 5,
        log_every: int = 1,
    ) -> List[Dict[str, float]]:
        """Full training loop with early stopping.

        Returns
        -------
        Training history (list of per-epoch metric dicts).
        """
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            epoch_metrics = {
                "epoch": epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **val_metrics,
                "time_s": time.time() - t0,
            }
            self.history.append(epoch_metrics)

            if epoch % log_every == 0:
                logger.info(
                    "Epoch %d/%d: train_loss=%.4f, recon=%.4f, vq=%.4f, "
                    "next_code=%.4f, perplexity=%.1f, usage=%.1f%%, lr=%.2e (%.1fs)",
                    epoch, max_epochs,
                    train_metrics["total_loss"],
                    train_metrics["recon_loss"],
                    train_metrics["vq_loss"],
                    train_metrics["next_code_loss"],
                    train_metrics["perplexity"],
                    train_metrics["codebook_usage"] * 100,
                    epoch_metrics["lr"],
                    epoch_metrics["time_s"],
                )

            # Early stopping on validation loss
            val_loss = val_metrics.get("val_total_loss")
            if val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if self.checkpoint_dir:
                        self.save_checkpoint(epoch, self.checkpoint_dir / "best.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        logger.info("Early stopping at epoch %d", epoch)
                        break

            # Periodic checkpoint
            if self.checkpoint_dir and checkpoint_every > 0 and epoch % checkpoint_every == 0:
                self.save_checkpoint(epoch, self.checkpoint_dir / f"epoch_{epoch}.pt")

        # Save final checkpoint and history
        if self.checkpoint_dir:
            self.save_checkpoint(epoch, self.checkpoint_dir / "final.pt")
            with open(self.checkpoint_dir / "history.json", "w") as f:
                json.dump(self.history, f, indent=2)

        return self.history


class StagedTrainer:
    """Progressive training: start with 1 file, gradually increase dataset size.

    Parameters
    ----------
    model:
        VQVAE model instance.
    hdf5_path:
        Path to preprocessed HDF5 file.
    stages:
        List of stage configs, each a dict with "name", "num_files", "max_epochs".
    trainer_kwargs:
        Additional kwargs passed to VQVAETrainer.
    """

    def __init__(
        self,
        model: VQVAE,
        hdf5_path: str,
        stages: List[Dict],
        trainer_kwargs: Optional[Dict] = None,
    ) -> None:
        self.model = model
        self.hdf5_path = hdf5_path
        self.stages = stages
        self.trainer_kwargs = trainer_kwargs or {}

    def run(self) -> List[Dict]:
        """Run all training stages sequentially."""
        from usv_language.src.data.dataset import (
            SpectrogramChunkDataset, chunk_collate_fn, create_splits,
        )
        import h5py

        with h5py.File(self.hdf5_path, "r") as f:
            all_stems = sorted(f.keys())

        all_history = []

        for stage in self.stages:
            name = stage["name"]
            num_files = stage.get("num_files")
            max_epochs = stage.get("max_epochs", 50)

            # Select files for this stage
            if num_files is not None and num_files < len(all_stems):
                stems = all_stems[:num_files]
            else:
                stems = all_stems

            logger.info("Stage '%s': %d files, %d epochs", name, len(stems), max_epochs)

            splits = create_splits(self.hdf5_path)
            train_stems = [s for s in splits["train"] if s in stems]
            val_stems = [s for s in splits["val"] if s in stems]

            if not train_stems:
                train_stems = stems
                val_stems = stems[:1] if len(stems) > 1 else None

            train_ds = SpectrogramChunkDataset(self.hdf5_path, file_stems=train_stems)
            train_loader = DataLoader(
                train_ds, batch_size=32, shuffle=True, collate_fn=chunk_collate_fn,
            )

            val_loader = None
            if val_stems:
                val_ds = SpectrogramChunkDataset(self.hdf5_path, file_stems=val_stems)
                val_loader = DataLoader(
                    val_ds, batch_size=32, shuffle=False, collate_fn=chunk_collate_fn,
                )

            checkpoint_dir = self.trainer_kwargs.get("checkpoint_dir")
            if checkpoint_dir:
                stage_dir = str(Path(checkpoint_dir) / name)
            else:
                stage_dir = None

            trainer = VQVAETrainer(
                model=self.model,
                train_loader=train_loader,
                val_loader=val_loader,
                checkpoint_dir=stage_dir,
                **{k: v for k, v in self.trainer_kwargs.items() if k != "checkpoint_dir"},
            )

            history = trainer.train(max_epochs=max_epochs)
            all_history.extend(history)

        return all_history
