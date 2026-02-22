"""CLI: Train VQ-VAE model.

Usage:
    python train.py --hdf5 <path> [--config configs/training.yaml] [--model-config configs/model.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
import torch

# Add project root so 'usv_language' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import usv_language.src  # noqa: F401

from usv_language.src.model.vqvae import VQVAE, VQVAEConfig
from usv_language.src.training.trainer import VQVAETrainer, StagedTrainer
from usv_language.src.data.dataset import (
    SpectrogramChunkDataset, chunk_collate_fn, create_splits,
)
from usv_language.src.data.normalization import compute_global_norm_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VQ-VAE model")
    parser.add_argument("--hdf5", type=str, required=True, help="Path to preprocessed HDF5")
    parser.add_argument("--config", type=str, default=None, help="Training config YAML")
    parser.add_argument("--model-config", type=str, default=None, help="Model config YAML")
    parser.add_argument("--data-config", type=str, default=None, help="Data config YAML")
    parser.add_argument("--device", type=str, default="cpu", help="Training device")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--staged", action="store_true", help="Use staged training")
    args = parser.parse_args()

    # Load configs
    model_cfg = {}
    if args.model_config:
        with open(args.model_config) as f:
            model_cfg = yaml.safe_load(f)

    train_cfg = {}
    if args.config:
        with open(args.config) as f:
            train_cfg = yaml.safe_load(f)

    data_cfg = {}
    if args.data_config:
        with open(args.data_config) as f:
            data_cfg = yaml.safe_load(f)

    # Determine n_freq from HDF5
    import h5py
    with h5py.File(args.hdf5, "r") as f:
        first_stem = list(f.keys())[0]
        n_freq = f[first_stem]["spectrogram"].shape[0]

    # Build model config
    transformer_cfg = model_cfg.get("transformer", {})
    encoder_cfg = model_cfg.get("encoder", {})
    decoder_cfg = model_cfg.get("decoder", {})
    vq_cfg = model_cfg.get("quantizer", {})

    vqvae_config = VQVAEConfig(
        n_freq=n_freq,
        d_model=transformer_cfg.get("d_model", 64),
        encoder_hidden_dims=tuple(encoder_cfg.get("hidden_dims", [256, 128])),
        decoder_hidden_dims=tuple(decoder_cfg.get("hidden_dims", [128, 256])),
        enc_dec_dropout=encoder_cfg.get("dropout", 0.1),
        codebook_size=vq_cfg.get("codebook_size", 512),
        ema_decay=vq_cfg.get("ema_decay", 0.99),
        commitment_weight=vq_cfg.get("commitment_weight", 0.25),
        dead_code_threshold=vq_cfg.get("dead_code_threshold", 2),
        n_heads=transformer_cfg.get("n_heads", 4),
        n_layers=transformer_cfg.get("n_layers", 4),
        d_ff=transformer_cfg.get("d_ff", 256),
        transformer_dropout=transformer_cfg.get("dropout", 0.1),
        max_seq_len=transformer_cfg.get("max_seq_len", 512),
    )

    model = VQVAE(vqvae_config)
    logger.info("Model: %d parameters", sum(p.numel() for p in model.parameters()))

    # Training config
    training = train_cfg.get("training", {})
    optimizer_cfg = train_cfg.get("optimizer", {})
    scheduler_cfg = train_cfg.get("scheduler", {})
    loss_cfg = train_cfg.get("loss_weights", {})
    dataset_cfg = data_cfg.get("dataset", {})

    trainer_kwargs = {
        "lr": optimizer_cfg.get("lr", 3e-4),
        "weight_decay": optimizer_cfg.get("weight_decay", 0.01),
        "gradient_clip_norm": training.get("gradient_clip_norm", 1.0),
        "warmup_steps": scheduler_cfg.get("warmup_steps", 500),
        "min_lr": scheduler_cfg.get("min_lr", 1e-6),
        "loss_weights": loss_cfg,
        "device": args.device,
        "checkpoint_dir": args.output_dir,
        "early_stopping_patience": training.get("early_stopping_patience", 10),
    }

    if args.staged:
        stages = train_cfg.get("staged_training", {}).get("stages", [
            {"name": "single_file", "num_files": 1, "max_epochs": 20},
        ])
        staged = StagedTrainer(model, args.hdf5, stages, trainer_kwargs)
        history = staged.run()
    else:
        # Standard training
        norm_params = compute_global_norm_params(args.hdf5)
        chunk_length = dataset_cfg.get("chunk_length", 256)
        batch_size = dataset_cfg.get("batch_size", 32)

        splits = create_splits(args.hdf5)
        train_ds = SpectrogramChunkDataset(
            args.hdf5, chunk_length=chunk_length,
            file_stems=splits["train"],
            norm_params=(norm_params.min_val, norm_params.max_val),
        )
        val_ds = SpectrogramChunkDataset(
            args.hdf5, chunk_length=chunk_length,
            file_stems=splits["val"],
            norm_params=(norm_params.min_val, norm_params.max_val),
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn=chunk_collate_fn,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, collate_fn=chunk_collate_fn,
        )

        trainer = VQVAETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            **trainer_kwargs,
        )

        if args.resume:
            start_epoch = trainer.load_checkpoint(args.resume)
            logger.info("Resumed from epoch %d", start_epoch)

        max_epochs = training.get("max_epochs", 100)
        history = trainer.train(max_epochs=max_epochs)

    logger.info("Training complete. %d epochs.", len(history))


if __name__ == "__main__":
    main()
