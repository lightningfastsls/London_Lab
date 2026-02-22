"""Phase 2 validation: Train encoder→decoder (no VQ, no transformer) on real data.

Sweeps d_model dimensions and plots reconstruction MSE.
Usage:
    python validate_autoencoder.py --hdf5 <path> [--epochs 20] [--d-models 32,64,128]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root so 'usv_language' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import usv_language.src  # noqa: F401

from usv_language.src.model.encoder import ColumnEncoder
from usv_language.src.model.decoder import ColumnDecoder
from usv_language.src.data.dataset import SpectrogramChunkDataset, chunk_collate_fn
from usv_language.src.data.normalization import compute_global_norm_params


def train_autoencoder(
    hdf5_path: str,
    d_model: int,
    n_freq: int,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 3e-4,
) -> dict:
    """Train a simple autoencoder and return loss history."""
    norm_params = compute_global_norm_params(hdf5_path)
    dataset = SpectrogramChunkDataset(
        hdf5_path,
        chunk_length=256,
        norm_params=(norm_params.min_val, norm_params.max_val),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=chunk_collate_fn)

    encoder = ColumnEncoder(n_freq, d_model)
    decoder = ColumnDecoder(d_model, n_freq)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            x = batch["spectrogram"]  # (B, seq_len, n_freq)
            z = encoder(x)
            x_hat = decoder(z)
            loss = criterion(x_hat, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        losses.append(avg_loss)
        print(f"  d_model={d_model}, epoch {epoch+1}/{epochs}, MSE={avg_loss:.6f}")

    return {"d_model": d_model, "losses": losses, "final_mse": losses[-1]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate autoencoder reconstruction")
    parser.add_argument("--hdf5", type=str, required=True, help="Path to preprocessed HDF5")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs per d_model")
    parser.add_argument("--d-models", type=str, default="32,64,128", help="Comma-separated d_model values")
    parser.add_argument("--output", type=str, default="ae_validation.png", help="Output plot path")
    args = parser.parse_args()

    import h5py
    with h5py.File(args.hdf5, "r") as f:
        first_stem = list(f.keys())[0]
        n_freq = f[first_stem]["spectrogram"].shape[0]

    d_models = [int(x) for x in args.d_models.split(",")]
    results = []
    for d in d_models:
        print(f"\nTraining d_model={d}:")
        result = train_autoencoder(args.hdf5, d, n_freq, epochs=args.epochs)
        results.append(result)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in results:
        ax.plot(r["losses"], label=f'd_model={r["d_model"]} (final={r["final_mse"]:.4f})')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Autoencoder Reconstruction: d_model Sweep")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {args.output}")


if __name__ == "__main__":
    main()
