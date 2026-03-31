#!/usr/bin/env python3
"""Fit temperature scaling on validation set for CNN probability calibration.

Usage:
    python scripts/calibrate_temperature.py \
        --model models/matched_windows/best_model.pt \
        --val-csv data/training/matched_windows/val.csv \
        --output models/matched_windows/temperature.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.models.cnn_classifier import USVClassifierCNN
from usv_spectrogram.models.data_loader import USVDataset
from usv_spectrogram.models.evaluate import load_model_checkpoint
from usv_spectrogram.postprocessing.calibration import (
    TemperatureScaler,
    compute_ece,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit temperature scaling on validation data."
    )
    parser.add_argument(
        "--model", type=Path, required=True,
        help="Path to trained CNN checkpoint (.pt file)",
    )
    parser.add_argument(
        "--val-csv", type=Path, required=True,
        help="Path to validation split CSV (candidate_id, spectrogram_path, label)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path for temperature JSON (default: alongside model)",
    )
    parser.add_argument(
        "--n-bins", type=int, default=15,
        help="Number of ECE bins (default: 15)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="DataLoader batch size (default: 64)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = args.model.parent / "temperature.json"

    # Load model
    print(f"Loading model from {args.model}")
    model, checkpoint = load_model_checkpoint(
        args.model, USVClassifierCNN, device="cpu"
    )
    device = next(model.parameters()).device

    # Load validation data
    print(f"Loading validation data from {args.val_csv}")
    val_dataset = USVDataset(args.val_csv)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Collect all logits and labels
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for spectrograms, labels in val_loader:
            spectrograms = spectrograms.to(device)
            logits = model.forward(spectrograms).squeeze(dim=1)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    print(f"Collected {len(logits)} validation samples")

    # Compute pre-calibration ECE
    raw_probs = 1.0 / (1.0 + np.exp(-logits))
    ece_before = compute_ece(raw_probs, labels, n_bins=args.n_bins)

    # Fit temperature
    scaler = TemperatureScaler()
    optimal_t = scaler.fit(logits, labels)

    # Compute post-calibration ECE
    calibrated_probs = scaler.calibrate(logits)
    ece_after = compute_ece(calibrated_probs, labels, n_bins=args.n_bins)

    # Summary
    print("\n=== Temperature Scaling Results ===")
    print(f"Optimal temperature: {optimal_t:.4f}")
    print(f"NLL before:  {scaler.nll_before:.4f}")
    print(f"NLL after:   {scaler.nll_after:.4f}")
    print(f"ECE before:  {ece_before:.4f}")
    print(f"ECE after:   {ece_after:.4f}")
    print(f"NLL change:  {scaler.nll_after - scaler.nll_before:+.4f}")
    print(f"ECE change:  {ece_after - ece_before:+.4f}")

    # Save
    scaler.save(args.output)
    print(f"\nSaved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
