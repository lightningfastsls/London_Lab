"""CLI entry point for null model statistical comparison analysis.

Loads a VQ-VAE checkpoint + hidden states, extracts code sequences,
then compares 7 information-theoretic metrics against a hierarchy of
null model surrogates.  Produces:
    - comparison_table.csv   (flat: one row per MetricComparison)
    - comparison_table.md    (publishable markdown table)
    - summary.txt            (plain-language interpretation)
    - raw_results.json       (all metric values for reproducibility)
    - metric_distributions/  (histograms for significant comparisons)

Usage:
    python run_null_model_analysis.py \\
        --hidden-states hidden_states_layer4.npy \\
        --vqvae-checkpoint vqvae_best.pt \\
        --output-dir null_model_output \\
        [--n-surrogates 100] [--seed 42] [--metadata metadata.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Bootstrap path so usv_language is importable when run as a script
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from usv_language.analysis.statistical_tests import (
    NullModelComparison,
    FullAnalysisResult,
)
from usv_language.analysis.null_models import NullModelConfig
from usv_language.analysis.sequence_analysis import extract_code_sequences
from usv_language.models.vqvae import HiddenStateVQVAE, VQVAEConfig


def load_vqvae(
    checkpoint_path: str, device: torch.device
) -> HiddenStateVQVAE:
    """Load a VQ-VAE checkpoint and return the model in eval mode."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", VQVAEConfig())
    model = HiddenStateVQVAE(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def write_csv(result: FullAnalysisResult, path: Path) -> None:
    """Write flat CSV with one row per MetricComparison."""
    fieldnames = [
        "metric_name", "null_model", "real_value", "null_mean", "null_std",
        "z_score", "rank_p_value", "effect_size", "n_surrogates", "significant",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in result.comparisons:
            writer.writerow({
                "metric_name": c.metric_name,
                "null_model": c.null_model,
                "real_value": f"{c.real_value:.6f}",
                "null_mean": f"{c.null_mean:.6f}",
                "null_std": f"{c.null_std:.6f}",
                "z_score": f"{c.z_score:.4f}",
                "rank_p_value": f"{c.rank_p_value:.4f}",
                "effect_size": f"{c.effect_size:.4f}",
                "n_surrogates": c.n_surrogates,
                "significant": c.significant,
            })


def write_raw_json(result: FullAnalysisResult, path: Path) -> None:
    """Write all metric values as JSON for reproducibility."""
    data = {
        "sequence_length": result.sequence_length,
        "codebook_size": result.codebook_size,
        "metrics_used": result.metrics_used,
        "null_models_used": result.null_models_used,
        "comparisons": [
            {
                "metric_name": c.metric_name,
                "null_model": c.null_model,
                "real_value": c.real_value,
                "null_mean": c.null_mean,
                "null_std": c.null_std,
                "z_score": c.z_score,
                "rank_p_value": c.rank_p_value,
                "effect_size": c.effect_size,
                "n_surrogates": c.n_surrogates,
                "significant": c.significant,
            }
            for c in result.comparisons
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_default(obj: object) -> object:
    """Handle NaN/inf serialization."""
    if isinstance(obj, float):
        if np.isnan(obj):
            return "NaN"
        if np.isinf(obj):
            return "Inf"
    return obj


def plot_significant_histograms(
    result: FullAnalysisResult,
    output_dir: Path,
) -> None:
    """Plot null distribution histograms for significant comparisons.

    For each significant (metric, null_model) pair:
      - Histogram of null surrogate values
      - Vertical red line at the real value
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist_dir = output_dir / "metric_distributions"
    hist_dir.mkdir(exist_ok=True)

    for c in result.comparisons:
        if not c.significant:
            continue

        fig, ax = plt.subplots(figsize=(7, 4))

        # We need null values — recompute or infer from mean/std
        # Since we don't store raw null values in MetricComparison,
        # we draw from the reported distribution for visualization.
        # This is a visual approximation using the normal parametric summary.
        if c.null_std > 0 and c.n_surrogates > 0:
            rng = np.random.RandomState(42)
            synthetic_nulls = rng.normal(
                c.null_mean, c.null_std, size=c.n_surrogates,
            )
            ax.hist(
                synthetic_nulls, bins=20, alpha=0.7,
                color="steelblue", edgecolor="white",
                label=f"Null ({c.null_model})",
            )
        else:
            ax.axhline(y=1, color="steelblue", alpha=0.5, label="Null (no variance)")

        ax.axvline(
            x=c.real_value, color="red", linewidth=2,
            linestyle="--", label=f"Real = {c.real_value:.4f}",
        )

        ax.set_title(
            f"{c.metric_name} vs {c.null_model} "
            f"(z={c.z_score:+.1f}, p={c.rank_p_value:.3f})"
        )
        ax.set_xlabel(c.metric_name)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.text(
            0.02, 0.95,
            "Approx: Normal(mean, std)\n(raw null values not stored)",
            transform=ax.transAxes, fontsize=7, va="top", color="gray",
        )
        fig.tight_layout()

        filename = f"{c.metric_name}_vs_{c.null_model}.png"
        fig.savefig(str(hist_dir / filename), dpi=150, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Null model statistical comparison for VQ-VAE code sequences",
    )
    parser.add_argument(
        "--hidden-states", required=True,
        help="Path to hidden states .npy file (N, d_model)",
    )
    parser.add_argument(
        "--vqvae-checkpoint", required=True,
        help="Path to VQ-VAE .pt checkpoint",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--metadata", default=None,
        help="Path to metadata.json (optional, for burstiness)",
    )
    parser.add_argument(
        "--n-surrogates", type=int, default=100,
        help="Number of surrogates per null model (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for VQ-VAE inference (default: cpu)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Load VQ-VAE and extract codes
    print(f"Loading VQ-VAE from {args.vqvae_checkpoint}...")
    vqvae = load_vqvae(args.vqvae_checkpoint, device)
    K = vqvae.config.codebook_size
    print(f"  Codebook size: K={K}")

    print(f"Loading hidden states from {args.hidden_states}...")
    hidden_states = np.load(args.hidden_states)
    print(f"  Shape: {hidden_states.shape}")

    print("Extracting code sequences...")
    codes = extract_code_sequences(hidden_states, vqvae, device=device)
    print(f"  {len(codes)} codes, range [{codes.min()}, {codes.max()}]")

    # Load event_times from metadata if available
    event_times = None
    if args.metadata and Path(args.metadata).exists():
        print(f"Loading metadata from {args.metadata}...")
        with open(args.metadata) as f:
            metadata = json.load(f)
        if "event_times" in metadata:
            event_times = np.array(metadata["event_times"], dtype=np.float64)
            print(f"  {len(event_times)} event timestamps loaded")

    # Run analysis
    null_config = NullModelConfig(
        n_surrogates=args.n_surrogates,
        random_seed=args.seed,
    )
    print(f"\nRunning null model comparison ({args.n_surrogates} surrogates)...")
    comparator = NullModelComparison()
    result = comparator.full_analysis(codes, K, event_times, null_config)

    # Write outputs
    print("\nWriting outputs...")

    write_csv(result, output_dir / "comparison_table.csv")
    print(f"  comparison_table.csv")

    md_content = result.to_markdown()
    (output_dir / "comparison_table.md").write_text(md_content, encoding="utf-8")
    print(f"  comparison_table.md")

    (output_dir / "summary.txt").write_text(result.summary, encoding="utf-8")
    print(f"  summary.txt")

    write_raw_json(result, output_dir / "raw_results.json")
    print(f"  raw_results.json")

    # Histograms for significant comparisons
    n_sig = sum(1 for c in result.comparisons if c.significant)
    if n_sig > 0:
        print(f"  Plotting {n_sig} significant comparison histograms...")
        plot_significant_histograms(result, output_dir)
        print(f"  metric_distributions/")

    # Print summary
    print(f"\n{result.summary}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
