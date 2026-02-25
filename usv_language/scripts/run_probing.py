"""CLI entry point for probing analysis of transformer hidden states.

Loads a trained SpectrogramTransformer checkpoint, runs inference with
``return_hidden_states=True`` to collect all layer representations,
extracts acoustic properties from the spectrogram data, then runs
cross-validated probing experiments to determine which layers encode
which properties.

Outputs:
    - probing_heatmap_linear.png  (layers x properties, linear probe)
    - probing_heatmap_mlp.png     (layers x properties, MLP probe)
    - layer_comparison.png        (mean score per layer, both probes)
    - results.json                (all ProbingResult data)
    - probing_report.md           (human-readable summary)

Usage:
    python run_probing.py \\
        --spectrogram-data spectrograms.npy \\
        --transformer-checkpoint transformer_best.pt \\
        --output-dir probing_output \\
        [--probe-types linear mlp] [--n-folds 5] [--max-samples 0] \\
        [--seed 42] [--onsets onsets.npy] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Bootstrap path so usv_language is importable when run as a script
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from usv_language.analysis.acoustic_properties import (
    AcousticPropertyConfig,
    extract_all_properties,
)
from usv_language.analysis.probing import (
    ProbingConfig,
    ProbingAnalysisPipeline,
    plot_probing_heatmap,
    plot_layer_comparison,
)
from usv_language.models.transformer import SpectrogramTransformer, TransformerConfig


def load_transformer(
    checkpoint_path: str, device: torch.device,
) -> SpectrogramTransformer:
    """Load a transformer checkpoint and return the model in eval mode."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", TransformerConfig())
    model = SpectrogramTransformer(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def extract_hidden_states(
    model: SpectrogramTransformer,
    spec_data: np.ndarray,
    device: torch.device,
) -> list[np.ndarray]:
    """Run inference and collect hidden states from all layers.

    Parameters
    ----------
    model:
        Trained transformer in eval mode.
    spec_data:
        Spectrogram frames, shape ``(N, n_freq)`` or ``(1, N, n_freq)``.
    device:
        Compute device.

    Returns
    -------
    List of arrays, one per layer, each shape ``(N, d_model)``.
    """
    if spec_data.ndim == 2:
        spec_data = spec_data[np.newaxis, ...]  # (1, N, n_freq)

    x = torch.from_numpy(spec_data).float().to(device)

    with torch.no_grad():
        _, hidden_states = model(x, return_hidden_states=True)

    # hidden_states: list of (1, N, d_model) tensors
    return [h.squeeze(0).cpu().numpy() for h in hidden_states]


def write_results_json(
    results, output_path: Path,
) -> None:
    """Serialize all ProbingResult data to JSON."""
    data = []
    for r in results:
        data.append({
            "property_name": r.property_name,
            "layer": r.layer,
            "probe_type": r.probe_type,
            "task_type": r.task_type,
            "score": r.score,
            "score_std": r.score_std,
            "n_samples": r.n_samples,
            "fold_scores": list(r.fold_scores),
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probing analysis of transformer hidden states",
    )
    parser.add_argument(
        "--spectrogram-data", required=True,
        help="Path to spectrogram .npy file (n_freq, T) or (T, n_freq)",
    )
    parser.add_argument(
        "--transformer-checkpoint", required=True,
        help="Path to transformer .pt checkpoint",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--probe-types", nargs="+", default=["linear", "mlp"],
        help="Probe types to run (default: linear mlp)",
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of CV folds (default: 5)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Max samples (0 = all, default: 0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--onsets", default=None,
        help="Path to USV onset indices .npy file (optional)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for transformer inference (default: cpu)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Load transformer and spectrogram
    print(f"Loading transformer from {args.transformer_checkpoint}...")
    model = load_transformer(args.transformer_checkpoint, device)
    print(f"  Config: {model.config}")

    print(f"Loading spectrogram data from {args.spectrogram_data}...")
    spec_data = np.load(args.spectrogram_data)
    print(f"  Shape: {spec_data.shape}")

    # Determine orientation using the transformer's known n_freq config.
    # Property extraction expects (n_freq, T); transformer expects (1, T, n_freq).
    n_freq_expected = model.config.n_freq
    if spec_data.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram, got shape {spec_data.shape}")

    if spec_data.shape[0] == n_freq_expected:
        spec_for_props = spec_data  # already (n_freq, T)
        spec_for_model = spec_data.T[np.newaxis, ...]  # (1, T, n_freq)
    elif spec_data.shape[1] == n_freq_expected:
        spec_for_props = spec_data.T  # was (T, n_freq), now (n_freq, T)
        spec_for_model = spec_data[np.newaxis, ...]  # (1, T, n_freq)
    else:
        raise ValueError(
            f"Cannot determine spectrogram orientation: shape {spec_data.shape}, "
            f"expected one dimension to be n_freq={n_freq_expected}"
        )

    T = spec_for_props.shape[1]
    print(f"  {T} frames, {spec_for_props.shape[0]} freq bins")

    # Extract hidden states
    print("Extracting hidden states from all layers...")
    x = torch.from_numpy(spec_for_model).float().to(device)
    with torch.no_grad():
        _, hidden_list = model(x, return_hidden_states=True)
    hidden_states = [h.squeeze(0).cpu().numpy() for h in hidden_list]
    n_layers = len(hidden_states)
    print(f"  {n_layers} layers, d_model={hidden_states[0].shape[1]}")

    # Extract acoustic properties
    print("Extracting acoustic properties...")
    onsets = None
    if args.onsets and Path(args.onsets).exists():
        onsets = np.load(args.onsets)
        print(f"  Loaded {len(onsets)} USV onsets")

    prop_config = AcousticPropertyConfig(n_freq_bins=spec_for_props.shape[0])
    properties = extract_all_properties(spec_for_props, prop_config, onsets)
    print(f"  Properties: {list(properties.keys())}")

    # Run probing analysis
    probing_config = ProbingConfig(
        probe_types=tuple(args.probe_types),
        n_folds=args.n_folds,
        max_samples=args.max_samples,
        random_seed=args.seed,
    )
    print(f"\nRunning probing analysis ({probing_config.n_folds}-fold CV)...")
    pipeline = ProbingAnalysisPipeline(probing_config)
    result = pipeline.run(hidden_states, properties)

    # Write outputs
    print("\nWriting outputs...")

    write_results_json(result.results, output_dir / "results.json")
    print("  results.json")

    (output_dir / "probing_report.md").write_text(
        result.summary, encoding="utf-8",
    )
    print("  probing_report.md")

    import matplotlib.pyplot as plt

    for pt in args.probe_types:
        heatmap_path = output_dir / f"probing_heatmap_{pt}.png"
        fig = plot_probing_heatmap(result, pt, output_path=str(heatmap_path))
        plt.close(fig)
        print(f"  probing_heatmap_{pt}.png")

    fig = plot_layer_comparison(result, output_path=str(output_dir / "layer_comparison.png"))
    plt.close(fig)
    print("  layer_comparison.png")

    # Print summary
    print(f"\n{result.summary}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
