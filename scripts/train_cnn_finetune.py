#!/usr/bin/env python3
"""Path C: fine-tune the production CNN from a checkpoint with optional
backbone freezing. Used for runs 4 and 5 of the lab fine-tune experiment.

Differences vs scripts/train_cnn.py:
  - Loads state dict from --init-from (production checkpoint)
  - Optionally freezes conv backbone (--freeze-mode {all,partial,none})
  - Lower default learning rate (1e-4 vs 1e-3) — fine-tunes need it
  - Optimizer rebuilt over only trainable params (so AdamW weight decay
    doesn't slowly shrink frozen weights toward zero)

Usage:
    .venv/bin/python scripts/train_cnn_finetune.py \
        --init-from models/hard_neg_retrain/best_model.pt \
        --train-csv data/training/lab_finetune_v1/csv/train.csv \
        --val-csv   data/training/lab_finetune_v1/csv/val.csv \
        --output-dir models/lab_finetune_v1_run4_pathC/ \
        --freeze-mode all \
        --num-epochs 30 --patience 8 \
        --learning-rate 0.0001 \
        --use-class-weights

freeze-mode:
  all     - freeze entire model.features; train only model.classifier (Run 4)
  partial - freeze all of features EXCEPT the last conv block; train it + classifier (Run 5)
  none    - full fine-tune; everything trains (sanity baseline)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.models import (
    USVClassifierCNN,
    Trainer,
    create_data_loaders,
)


def load_checkpoint_and_build_model(
    path: Path, dropout_rate: float = 0.5
) -> tuple[USVClassifierCNN, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    num_filters = ckpt.get("num_filters", [32, 96, 192])
    dense_units = ckpt.get("dense_units", 64)
    model = USVClassifierCNN(
        num_filters=num_filters,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch')}, "
          f"num_filters={num_filters}, dense_units={dense_units}")
    return model, ckpt


def freeze_backbone(model: USVClassifierCNN, mode: str) -> None:
    """Apply freeze-mode to model in place.

    The architecture is 3 conv blocks in model.features, each block is
    4 nn modules (Conv2d, BatchNorm2d, ReLU, MaxPool2d). So:
        block 0: features[0..3]
        block 1: features[4..7]
        block 2: features[8..11]  <- the "last conv block"
    """
    if mode == "none":
        return
    feat = model.features
    if mode == "all":
        for p in feat.parameters():
            p.requires_grad = False
        return
    if mode == "partial":
        # Freeze first two blocks (indices 0..7), leave last block (8..11) trainable
        for i, layer in enumerate(feat):
            if i < 8:
                for p in layer.parameters():
                    p.requires_grad = False
        return
    raise ValueError(f"Unknown freeze-mode: {mode}")


def summarize_params(model: torch.nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"Total params:     {total:>10,}")
    print(f"Trainable params: {trainable:>10,} ({100*trainable/total:.1f}%)")
    print(f"Frozen params:    {frozen:>10,} ({100*frozen/total:.1f}%)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--init-from", type=Path, required=True,
                   help="Production checkpoint to initialize from")
    p.add_argument("--train-csv", type=Path, required=True)
    p.add_argument("--val-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--freeze-mode", choices=["all", "partial", "none"], default="all")

    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=0.0001,
                   help="Lower than from-scratch default (was 0.001).")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--dropout-rate", type=float, default=0.5)
    p.add_argument("--use-class-weights", action="store_true", default=False)
    p.add_argument("--normalize-mode", default="per_image",
                   choices=["per_image", "global"])
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if not args.init_from.exists():
        print(f"ERROR: --init-from not found: {args.init_from}", file=sys.stderr)
        return 1
    if not args.train_csv.exists():
        print(f"ERROR: --train-csv not found: {args.train_csv}", file=sys.stderr)
        return 1
    if not args.val_csv.exists():
        print(f"ERROR: --val-csv not found: {args.val_csv}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LAB FINE-TUNE — Path C (frozen-backbone variant)")
    print("=" * 80)
    print(f"Init from:        {args.init_from}")
    print(f"Train CSV:        {args.train_csv}")
    print(f"Val CSV:          {args.val_csv}")
    print(f"Output dir:       {args.output_dir}")
    print(f"Freeze mode:      {args.freeze_mode}")
    print(f"Learning rate:    {args.learning_rate}")
    print(f"Weight decay:     {args.weight_decay}")
    print(f"Num epochs (max): {args.num_epochs}")
    print(f"Patience:         {args.patience}")
    print(f"Dropout:          {args.dropout_rate}")
    print(f"Class weights:    {args.use_class_weights}")
    print(f"Normalize:        {args.normalize_mode}")
    print(f"Seed:             {args.seed}")
    print("=" * 80)

    print("\nLoading checkpoint + building model...")
    model, ckpt = load_checkpoint_and_build_model(
        args.init_from, dropout_rate=args.dropout_rate
    )

    print(f"\nApplying freeze-mode={args.freeze_mode}...")
    freeze_backbone(model, args.freeze_mode)
    summarize_params(model)

    print("\nLoading data...")
    train_loader, val_loader, class_weights = create_data_loaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        batch_size=args.batch_size,
        num_workers=2,
        normalize_mode=args.normalize_mode,
        use_class_weights=args.use_class_weights,
    )

    print("\nConstructing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights if args.use_class_weights else None,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        checkpoint_dir=args.output_dir,
        device=args.device,
    )

    # Replace optimizer with one that only includes trainable params, so
    # frozen-backbone weights aren't slowly weight-decayed toward zero.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        print("ERROR: no trainable parameters — check --freeze-mode", file=sys.stderr)
        return 1
    trainer.optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer, mode="min", factor=0.5, patience=5,
    )
    n_trainable_groups = len(trainer.optimizer.param_groups[0]["params"])
    print(f"Optimizer rebuilt over {n_trainable_groups} trainable tensors.")

    print("\nStarting fine-tune...\n")
    history = trainer.train(num_epochs=args.num_epochs, verbose=True)

    print()
    print("=" * 80)
    print("FINE-TUNE COMPLETE")
    print("=" * 80)
    print(f"Best val loss:  {min(history.val_loss):.4f}")
    print(f"Best val acc:   {max(history.val_acc):.4f}")
    print(f"Best val F1:    {max(history.val_f1):.4f}")
    print(f"Checkpoints:    {args.output_dir}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
