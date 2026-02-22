"""Multi-layer VQ-VAE comparison: trains on different transformer layers.

Trains a VQ-VAE on hidden states extracted from layers 2, 4, 6, 8 of the
autoregressive transformer, then generates a markdown report comparing
reconstruction quality, codebook utilization, and perplexity.

Usage:
    python -m usv_language.training.compare_layers \
        --hidden-states-dir /path/to/extracted_states \
        --output-dir /path/to/comparison_results

Expects hidden state files named: hidden_states_layer{N}.npy
Each file has shape (total_frames, 512).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Bootstrap: ensure project root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from usv_language.training.train_vqvae import parse_args as parse_vqvae_args
from usv_language.training.train_vqvae import train as train_vqvae

logger = logging.getLogger(__name__)

DEFAULT_LAYERS = [2, 4, 6, 8]


def score_layer(metrics: dict, codebook_size: int = 64) -> float:
    """Compute weighted score for a layer's VQ-VAE metrics.

    Scoring: 40% perplexity + 30% utilization + 30% (1 - recon_loss).
    Higher is better. Perplexity is normalized by codebook_size.

    Parameters
    ----------
    metrics:
        Dict with ``perplexity``, ``codebook_usage``, ``recon_loss``.
    codebook_size:
        K value used for perplexity normalization.

    Returns
    -------
    Score in [0, 1] range (approximately).
    """
    norm_perplexity = min(metrics.get("perplexity", 1.0) / codebook_size, 1.0)
    utilization = metrics.get("codebook_usage", 0.0)
    recon_quality = max(1.0 - metrics.get("recon_loss", 1.0), 0.0)

    return 0.4 * norm_perplexity + 0.3 * utilization + 0.3 * recon_quality


def generate_report(
    results: dict[int, dict],
    output_path: Path,
    codebook_size: int = 64,
) -> None:
    """Generate a markdown comparison report.

    Parameters
    ----------
    results:
        Mapping from layer number to final validation metrics dict.
    output_path:
        Path for the output markdown file.
    codebook_size:
        K value for perplexity normalization in scoring.
    """
    lines = [
        "# VQ-VAE Layer Comparison Report",
        "",
        "Trains a VQ-VAE (K=64, codebook_dim=64) on transformer hidden states",
        "from different layers to discover which layer's representations are",
        "most amenable to discrete codebook compression.",
        "",
        "## Metrics Table",
        "",
        "| Layer | Recon Loss | Commit Loss | Perplexity | Utilization | Score |",
        "|------:|-----------:|------------:|-----------:|------------:|------:|",
    ]

    scored = {}
    for layer in sorted(results.keys()):
        m = results[layer]
        s = score_layer(m, codebook_size=codebook_size)
        scored[layer] = s
        lines.append(
            f"| {layer:5d} "
            f"| {m.get('recon_loss', float('nan')):10.4f} "
            f"| {m.get('commit_loss', float('nan')):11.4f} "
            f"| {m.get('perplexity', float('nan')):10.1f} "
            f"| {m.get('codebook_usage', float('nan')) * 100:10.1f}% "
            f"| {s:5.3f} |"
        )

    lines.append("")

    # Recommendation
    if scored:
        best_layer = max(scored, key=scored.get)
        lines.extend([
            "## Recommendation",
            "",
            f"**Layer {best_layer}** achieves the highest weighted score "
            f"({scored[best_layer]:.3f}).",
            "",
            "Scoring weights: 40% perplexity (codebook diversity), "
            "30% utilization (codes used), 30% reconstruction quality.",
            "",
        ])

    # Interpretation guide
    lines.extend([
        "## Interpretation Guide",
        "",
        "- **Recon Loss**: Lower = hidden states are more compressible "
        "into discrete codes",
        "- **Perplexity**: Higher = more diverse codebook usage "
        "(ideal: close to K=64)",
        "- **Utilization**: Fraction of codebook entries actively used "
        "(target: >50%)",
        "- **Score**: Weighted combination — higher is better",
        "",
        "Higher layers tend to encode more abstract, compositional features.",
        "Lower layers encode more local, acoustic features.",
        "The best layer for interpretability balances compressibility "
        "with codebook diversity.",
        "",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to %s", output_path)


def compare_layers(args: argparse.Namespace) -> None:
    """Train VQ-VAE on each specified layer and generate comparison report."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    hs_dir = Path(args.hidden_states_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = args.layers
    results: dict[int, dict] = {}

    for layer in layers:
        hs_path = hs_dir / f"hidden_states_layer{layer}.npy"
        if not hs_path.exists():
            logger.warning("Skipping layer %d: %s not found", layer, hs_path)
            continue

        layer_dir = output_dir / f"layer_{layer}"
        logger.info("=== Training VQ-VAE on layer %d ===", layer)

        # Build args for train_vqvae
        train_argv = [
            "--hidden-states", str(hs_path),
            "--output-dir", str(layer_dir),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--patience", str(args.patience),
            "--d-model", str(args.d_model),
            "--codebook-size", str(args.codebook_size),
            "--codebook-dim", str(args.codebook_dim),
            "--num-workers", str(args.num_workers),
            "--seed", str(args.seed),
        ]
        vqvae_args = parse_vqvae_args(train_argv)
        metrics = train_vqvae(vqvae_args)

        if metrics:
            results[layer] = metrics
            logger.info(
                "Layer %d: recon=%.4f, perplexity=%.1f, usage=%.1f%%",
                layer,
                metrics.get("recon_loss", float("nan")),
                metrics.get("perplexity", float("nan")),
                metrics.get("codebook_usage", 0) * 100,
            )
        else:
            logger.warning("Layer %d: no metrics returned", layer)

    # Generate report
    report_path = output_dir / "comparison_report.md"
    generate_report(results, report_path, codebook_size=args.codebook_size)

    logger.info("Comparison complete. Report: %s", report_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Compare VQ-VAE training across transformer layers",
    )

    p.add_argument("--hidden-states-dir", type=str, required=True,
                   help="Directory with hidden_states_layer{N}.npy files")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Directory for comparison results")

    # Layers
    p.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS,
                   help="Layer indices to compare (default: 2 4 6 8)")

    # Training overrides (passed to train_vqvae)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--codebook-size", type=int, default=64)
    p.add_argument("--codebook-dim", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args(argv)


if __name__ == "__main__":
    compare_layers(parse_args())
