"""Phase 3 validation: Train encoder→VQ→decoder on real data.

Monitors codebook utilization, perplexity, and reconstruction quality.
Usage:
    python validate_vq.py --hdf5 <path> [--epochs 30] [--d-model 64]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root so 'usv_language' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import usv_language.src  # noqa: F401

from usv_language.src.model.encoder import ColumnEncoder
from usv_language.src.model.decoder import ColumnDecoder
from usv_language.src.model.quantizer import VectorQuantizer
from usv_language.src.data.dataset import SpectrogramChunkDataset, chunk_collate_fn
from usv_language.src.data.normalization import compute_global_norm_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate VQ-AE training")
    parser.add_argument("--hdf5", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--codebook-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=str, default="vq_validation.png")
    args = parser.parse_args()

    import h5py
    with h5py.File(args.hdf5, "r") as f:
        first_stem = list(f.keys())[0]
        n_freq = f[first_stem]["spectrogram"].shape[0]

    norm_params = compute_global_norm_params(args.hdf5)
    dataset = SpectrogramChunkDataset(
        args.hdf5, chunk_length=256,
        norm_params=(norm_params.min_val, norm_params.max_val),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=chunk_collate_fn)

    encoder = ColumnEncoder(n_freq, args.d_model)
    quantizer = VectorQuantizer(args.codebook_size, args.d_model)
    decoder = ColumnDecoder(args.d_model, n_freq)

    params = list(encoder.parameters()) + list(quantizer.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-4)
    recon_criterion = nn.MSELoss()

    history = {"recon_loss": [], "vq_loss": [], "perplexity": [], "codebook_usage": []}

    for epoch in range(args.epochs):
        epoch_recon = epoch_vq = epoch_perp = epoch_usage = 0.0
        n_batches = 0

        for batch in loader:
            x = batch["spectrogram"]
            z = encoder(x)
            vq_out = quantizer(z)
            x_hat = decoder(vq_out["z_q"])

            recon_loss = recon_criterion(x_hat, x)
            loss = recon_loss + vq_out["vq_loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_recon += recon_loss.item()
            epoch_vq += vq_out["vq_loss"].item()
            epoch_perp += vq_out["perplexity"].item()
            epoch_usage += vq_out["codebook_usage"].item()
            n_batches += 1

        for key, val in [
            ("recon_loss", epoch_recon),
            ("vq_loss", epoch_vq),
            ("perplexity", epoch_perp),
            ("codebook_usage", epoch_usage),
        ]:
            history[key].append(val / max(1, n_batches))

        print(
            f"Epoch {epoch+1}/{args.epochs}: "
            f"recon={history['recon_loss'][-1]:.4f}, "
            f"vq={history['vq_loss'][-1]:.4f}, "
            f"perplexity={history['perplexity'][-1]:.1f}, "
            f"usage={history['codebook_usage'][-1]:.2%}"
        )

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, title) in zip(axes.ravel(), [
        ("recon_loss", "Reconstruction MSE"),
        ("vq_loss", "VQ Commitment Loss"),
        ("perplexity", f"Perplexity (of {args.codebook_size})"),
        ("codebook_usage", "Codebook Usage Fraction"),
    ]):
        ax.plot(history[key])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"VQ-AE Validation (d_model={args.d_model}, K={args.codebook_size})")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {args.output}")


if __name__ == "__main__":
    main()
