"""CLI: Analyze trained VQ-VAE checkpoint.

Usage:
    python analyze.py --checkpoint <path.pt> --hdf5 <path.h5> --output-dir <dir>
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root so 'usv_language' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import usv_language.src  # noqa: F401

from usv_language.src.model.vqvae import VQVAE, VQVAEConfig
from usv_language.src.data.dataset import SpectrogramChunkDataset, chunk_collate_fn
from usv_language.src.data.normalization import compute_global_norm_params
from usv_language.src.analysis.codebook_viz import (
    decode_codebook_entries, plot_codebook_profiles, plot_codebook_usage,
)
from usv_language.src.analysis.sequence_analysis import (
    compute_transition_matrix, transition_entropy, top_ngrams,
    mutual_information_at_lag, entropy_rate, plot_transition_matrix,
)
from usv_language.src.analysis.generative_probing import round_trip_test, plot_round_trip

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze trained VQ-VAE")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--hdf5", type=str, required=True, help="Preprocessed HDF5 path")
    parser.add_argument("--output-dir", type=str, default="analysis_output", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = VQVAE(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Loaded model from %s", args.checkpoint)

    # 1. Codebook visualization
    logger.info("Decoding codebook entries...")
    profiles = decode_codebook_entries(model)
    plot_codebook_profiles(profiles, output_path=str(output_dir / "codebook_profiles.png"))
    plot_codebook_usage(model, output_path=str(output_dir / "codebook_usage.png"))
    logger.info("Codebook plots saved.")

    # 2. Encode dataset to codes
    logger.info("Encoding dataset to codes...")
    norm_params = compute_global_norm_params(args.hdf5)
    dataset = SpectrogramChunkDataset(
        args.hdf5, chunk_length=256,
        norm_params=(norm_params.min_val, norm_params.max_val),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, collate_fn=chunk_collate_fn,
    )

    all_codes = []
    for batch in loader:
        x = batch["spectrogram"]
        codes = model.encode_to_codes(x)
        all_codes.append(codes.cpu().numpy())

    all_codes = np.concatenate(all_codes).ravel()
    logger.info("Encoded %d code tokens", len(all_codes))

    # 3. Transition analysis
    logger.info("Computing transition matrix...")
    T = compute_transition_matrix(all_codes, config.codebook_size)
    plot_transition_matrix(T, output_path=str(output_dir / "transition_matrix.png"))

    t_entropy = transition_entropy(T)
    logger.info("Transition entropy: mean=%.2f, min=%.2f, max=%.2f",
                t_entropy.mean(), t_entropy.min(), t_entropy.max())

    # 4. N-grams
    for n in [2, 3, 4]:
        top = top_ngrams(all_codes, n, top_k=10)
        logger.info("Top %d-grams: %s", n, top[:5])

    # 5. Mutual information
    lags = [1, 2, 5, 10, 20]
    mi_values = [mutual_information_at_lag(all_codes, config.codebook_size, lag) for lag in lags]
    logger.info("MI at lags %s: %s", lags, [f"{v:.3f}" for v in mi_values])

    # 6. Entropy rate
    rates = entropy_rate(all_codes, max_order=5)
    logger.info("Entropy rates (order 1-5): %s", [f"{r:.3f}" for r in rates])

    # 7. Round-trip test
    batch = next(iter(loader))
    result = round_trip_test(model, batch["spectrogram"][:4])
    plot_round_trip(result, output_path=str(output_dir / "round_trip.png"))
    logger.info("Round-trip MSE: %.4f", result["mse"])

    logger.info("All analysis outputs saved to %s", output_dir)


if __name__ == "__main__":
    main()
