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
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Bootstrap: ensure project root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from usv_language.training.train_vqvae import parse_args as parse_vqvae_args
from usv_language.training.train_vqvae import train as train_vqvae

logger = logging.getLogger(__name__)

DEFAULT_LAYERS = [2, 4, 6, 8]


def load_extraction_metadata(hidden_states_dir: Path) -> dict | None:
    """Load extractor metadata when present."""
    metadata_path = hidden_states_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    with open(metadata_path, encoding="utf-8") as f:
        return json.load(f)


def validate_hidden_state_artifact(
    hidden_states_path: Path,
    layer: int,
    metadata: dict | None = None,
) -> None:
    """Validate a hidden-state artifact against metadata and basic invariants."""
    if not hidden_states_path.exists():
        raise FileNotFoundError(f"missing hidden state file: {hidden_states_path.name}")

    if metadata is not None:
        layers_extracted = metadata.get("layers_extracted")
        if layers_extracted and layer not in layers_extracted:
            raise ValueError(
                f"{hidden_states_path.name} requests layer {layer}, but metadata "
                f"lists layers_extracted={layers_extracted}"
            )

    hidden_states = np.load(str(hidden_states_path), mmap_mode="r")
    try:
        if hidden_states.ndim != 2:
            raise ValueError(
                f"{hidden_states_path.name} must have shape (frames, d_model), "
                f"got {hidden_states.shape}"
            )
        if hidden_states.shape[0] == 0:
            raise ValueError(f"{hidden_states_path.name} is empty")
        if hidden_states.shape[1] == 0:
            raise ValueError(f"{hidden_states_path.name} has zero feature width")

        if metadata is not None:
            expected_frames = metadata.get("total_frames")
            if expected_frames is not None and hidden_states.shape[0] != int(expected_frames):
                raise ValueError(
                    f"{hidden_states_path.name} frame count {hidden_states.shape[0]} does not "
                    f"match metadata total_frames={expected_frames}"
                )

            expected_width = metadata.get("d_model")
            if expected_width is not None and hidden_states.shape[1] != int(expected_width):
                raise ValueError(
                    f"{hidden_states_path.name} width {hidden_states.shape[1]} does not match "
                    f"metadata d_model={expected_width}"
                )
    finally:
        del hidden_states


def load_vqvae_run_config(layer_dir: Path) -> dict | None:
    """Load a layer's persisted VQ-VAE config for report provenance."""
    config_path = layer_dir / "config.json"
    if not config_path.exists():
        return None
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


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
    failures: dict[int, str] | None = None,
    extraction_metadata: dict | None = None,
    layer_run_configs: dict[int, dict] | None = None,
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
    failures:
        Optional mapping from layer number to failure/skip reason.
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
    elif failures:
        lines.extend([
            "## Recommendation",
            "",
            "No layer completed successfully. Inspect the failures below.",
            "",
        ])

    if failures:
        lines.extend([
            "## Failed / Skipped Layers",
            "",
        ])
        for layer in sorted(failures):
            lines.append(f"- Layer {layer}: {failures[layer]}")
        lines.append("")

    if extraction_metadata:
        lines.extend([
            "## Extraction Provenance",
            "",
            f"- Extractor checkpoint: `{extraction_metadata.get('checkpoint', 'unknown')}`",
            f"- Extractor checkpoint epoch: `{extraction_metadata.get('checkpoint_epoch', 'unknown')}`",
            f"- Layers extracted: `{extraction_metadata.get('layers_extracted', [])}`",
            f"- Primary layer: `{extraction_metadata.get('primary_layer', 'unknown')}`",
            f"- Total frames: `{extraction_metadata.get('total_frames', 'unknown')}`",
            f"- Hidden width: `{extraction_metadata.get('d_model', 'unknown')}`",
            "",
        ])

    if layer_run_configs:
        lines.extend([
            "## Layer Artifact Provenance",
            "",
        ])
        for layer in sorted(layer_run_configs):
            run_config = layer_run_configs[layer]
            provenance = run_config.get("provenance", {})
            dataset = run_config.get("dataset", {})
            training = run_config.get("training", {})
            lines.extend([
                f"### Layer {layer}",
                "",
                f"- Hidden states: `{provenance.get('hidden_states_path', dataset.get('hidden_states', 'unknown'))}`",
                f"- Metadata path: `{provenance.get('metadata_path', 'not captured')}`",
                f"- Source layer: `{provenance.get('source_layer', 'unknown')}`",
                f"- Window/stride: `{dataset.get('window_size', 'unknown')}` / `{dataset.get('stride', 'unknown')}`",
                f"- Epochs/patience: `{training.get('epochs', 'unknown')}` / `{training.get('patience', 'unknown')}`",
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
    metadata = load_extraction_metadata(hs_dir)

    layers = args.layers
    results: dict[int, dict] = {}
    failures: dict[int, str] = {}
    layer_run_configs: dict[int, dict] = {}

    for layer in layers:
        hs_path = hs_dir / f"hidden_states_layer{layer}.npy"
        try:
            validate_hidden_state_artifact(hs_path, layer, metadata=metadata)
        except Exception as exc:
            logger.warning("Skipping layer %d: %s", layer, exc)
            failures[layer] = str(exc)
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
            "--window-size", str(args.window_size),
            "--stride", str(args.stride),
            "--val-fraction", str(args.val_fraction),
            "--num-workers", str(args.num_workers),
            "--seed", str(args.seed),
        ]
        vqvae_args = parse_vqvae_args(train_argv)
        try:
            metrics = train_vqvae(vqvae_args)
        except Exception as exc:
            logger.exception("Layer %d failed: %s", layer, exc)
            failures[layer] = str(exc)
            continue

        if metrics:
            results[layer] = metrics
            run_config = load_vqvae_run_config(layer_dir)
            if run_config is not None:
                layer_run_configs[layer] = run_config
            logger.info(
                "Layer %d: recon=%.4f, perplexity=%.1f, usage=%.1f%%",
                layer,
                metrics.get("recon_loss", float("nan")),
                metrics.get("perplexity", float("nan")),
                metrics.get("codebook_usage", 0) * 100,
            )
        else:
            logger.warning("Layer %d: no metrics returned", layer)
            failures[layer] = "no metrics returned"

    # Generate report
    report_path = output_dir / "comparison_report.md"
    generate_report(
        results,
        report_path,
        codebook_size=args.codebook_size,
        failures=failures,
        extraction_metadata=metadata,
        layer_run_configs=layer_run_configs,
    )

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
    p.add_argument("--window-size", type=int, default=128)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args(argv)


if __name__ == "__main__":
    compare_layers(parse_args())
