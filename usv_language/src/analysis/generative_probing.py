"""Generative probing: round-trip tests, code-swap analysis, generation.

Tools for understanding model behavior by manipulating code sequences
and observing their effect on reconstructed spectrograms.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from usv_language.src.model.vqvae import VQVAE


def round_trip_test(
    model: VQVAE,
    x: torch.Tensor,
) -> dict:
    """Test spectrogram → codes → spectrogram reconstruction fidelity.

    Parameters
    ----------
    model:
        Trained VQVAE model.
    x:
        Input spectrogram (B, seq_len, n_freq).

    Returns
    -------
    Dict with: original, codes, reconstructed, mse.
    """
    model.eval()
    with torch.no_grad():
        codes = model.encode_to_codes(x)
        recon = model.decode_codes_to_spectrogram(codes)
        mse = torch.nn.functional.mse_loss(recon, x).item()

    return {
        "original": x.cpu().numpy(),
        "codes": codes.cpu().numpy(),
        "reconstructed": recon.cpu().numpy(),
        "mse": mse,
    }


def code_swap_analysis(
    model: VQVAE,
    codes: torch.LongTensor,
    position: int,
    new_code: int,
) -> dict:
    """Change one code and observe effect on reconstruction.

    Parameters
    ----------
    model: Trained VQVAE model.
    codes: (1, seq_len) code sequence.
    position: Index of code to swap.
    new_code: Replacement code index.

    Returns
    -------
    Dict with: original_recon, swapped_recon, diff, position, old_code, new_code.
    """
    model.eval()
    with torch.no_grad():
        original_recon = model.decode_codes_to_spectrogram(codes)

        swapped = codes.clone()
        old_code = codes[0, position].item()
        swapped[0, position] = new_code
        swapped_recon = model.decode_codes_to_spectrogram(swapped)

        diff = (swapped_recon - original_recon).abs()

    return {
        "original_recon": original_recon.cpu().numpy(),
        "swapped_recon": swapped_recon.cpu().numpy(),
        "diff": diff.cpu().numpy(),
        "position": position,
        "old_code": old_code,
        "new_code": new_code,
    }


def concept_scan(
    model: VQVAE,
    codes: torch.LongTensor,
    position: int,
) -> np.ndarray:
    """At a fixed context, try every possible code at one position.

    Parameters
    ----------
    model: Trained VQVAE model.
    codes: (1, seq_len) code sequence (context).
    position: Position to scan.

    Returns
    -------
    reconstructions: (codebook_size, n_freq) — decoded column at the scanned position
        for each possible code value.
    """
    model.eval()
    K = model.config.codebook_size
    results = []

    with torch.no_grad():
        for code_id in range(K):
            modified = codes.clone()
            modified[0, position] = code_id
            recon = model.decode_codes_to_spectrogram(modified)
            results.append(recon[0, position].cpu().numpy())

    return np.stack(results)  # (K, n_freq)


def plot_round_trip(
    result: dict,
    sample_idx: int = 0,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot original vs reconstructed spectrogram from round-trip test.

    Parameters
    ----------
    result: Output from round_trip_test().
    sample_idx: Which sample in batch to plot.
    output_path: If provided, save figure.

    Returns
    -------
    matplotlib Figure.
    """
    original = result["original"][sample_idx].T  # (n_freq, seq_len)
    reconstructed = result["reconstructed"][sample_idx].T

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(original, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title("Original")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Frequency")

    axes[1].imshow(reconstructed, aspect="auto", origin="lower", cmap="viridis")
    axes[1].set_title("Reconstructed")
    axes[1].set_xlabel("Time")

    diff = np.abs(original - reconstructed)
    axes[2].imshow(diff, aspect="auto", origin="lower", cmap="hot")
    axes[2].set_title(f"Absolute Diff (MSE={result['mse']:.4f})")
    axes[2].set_xlabel("Time")

    fig.suptitle("Round-Trip: Spectrogram → Codes → Spectrogram")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
