"""CLI entry point for the Phase 8.4 analysis suite.

Usage:
    python -m usv_language.analysis.run_analysis \\
        --transformer-checkpoint best.pt \\
        --vqvae-checkpoint vqvae_best.pt \\
        --hidden-states hidden_states_layer4.npy \\
        --output-dir analysis_output \\
        --source-layer 4

Orchestrates all five analysis modules:
1. Codebook visualization (profiles, usage, projection, exemplars)
2. Sequence analysis (Zipf, transitions, entropy, MI)
3. Concept manipulation (injection, scan)
4. Context analysis (group comparison, if metadata available)
5. Compositionality (bigram productivity, positional independence)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

from usv_language.analysis.config import AnalysisConfig
from usv_language.models.transformer import SpectrogramTransformer, TransformerConfig
from usv_language.models.vqvae import HiddenStateVQVAE, VQVAEConfig


def load_models(
    transformer_path: str,
    vqvae_path: str,
    device: torch.device,
) -> tuple:
    """Load frozen transformer and VQ-VAE from checkpoints.

    Returns
    -------
    (transformer, vqvae) tuple, both in eval mode on ``device``.
    """
    # Load transformer
    t_ckpt = torch.load(transformer_path, map_location=device, weights_only=False)
    t_config = t_ckpt.get("config", TransformerConfig())
    transformer = SpectrogramTransformer(t_config).to(device)
    transformer.load_state_dict(t_ckpt["model_state_dict"])
    transformer.eval()

    # Load VQ-VAE
    v_ckpt = torch.load(vqvae_path, map_location=device, weights_only=False)
    v_config = v_ckpt.get("config", VQVAEConfig())
    vqvae = HiddenStateVQVAE(v_config).to(device)
    vqvae.load_state_dict(v_ckpt["model_state_dict"])
    vqvae.eval()

    return transformer, vqvae


def run_analysis(
    transformer_path: str,
    vqvae_path: str,
    hidden_states_path: str,
    output_dir: str,
    source_layer: int = 4,
    metadata_path: str | None = None,
    device_name: str = "cpu",
) -> None:
    """Run the full analysis pipeline.

    Parameters
    ----------
    transformer_path:
        Path to transformer checkpoint (.pt).
    vqvae_path:
        Path to VQ-VAE checkpoint (.pt).
    hidden_states_path:
        Path to hidden states (.npy), shape (N, d_model).
    output_dir:
        Directory for output figures and reports.
    source_layer:
        1-indexed transformer layer for hidden states.
    metadata_path:
        Optional path to metadata.json.
    device_name:
        Device for inference.
    """
    from usv_language.analysis import codebook_viz
    from usv_language.analysis import sequence_analysis
    from usv_language.analysis import concept_manipulation
    from usv_language.analysis import context_analysis
    from usv_language.analysis import compositionality

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)

    config = AnalysisConfig(source_layer=source_layer)

    print(f"Loading models...")
    transformer, vqvae = load_models(transformer_path, vqvae_path, device)
    K = vqvae.config.codebook_size
    print(f"  Transformer: {transformer.count_parameters():,} params")
    print(f"  VQ-VAE: {vqvae.count_parameters():,} params, K={K}")

    print(f"Loading hidden states from {hidden_states_path}...")
    hidden_states = np.load(hidden_states_path)
    print(f"  Shape: {hidden_states.shape}")

    # ---------------------------------------------------------------
    # 1. Codebook visualization
    # ---------------------------------------------------------------
    print("\n=== Codebook Visualization ===")

    print("  Decoding all codebook entries...")
    profiles = codebook_viz.decode_all_entries(transformer, vqvae, source_layer)
    codebook_viz.plot_decoded_profiles(
        profiles, config, str(output_dir / "codebook_profiles.png"),
    )
    print(f"  Saved codebook_profiles.png")

    codebook_viz.plot_codebook_usage(vqvae, str(output_dir / "codebook_usage.png"))
    print(f"  Saved codebook_usage.png")

    codebook_viz.plot_codebook_projection(
        vqvae, profiles, config, str(output_dir / "codebook_projection.png"),
    )
    print(f"  Saved codebook_projection.png")

    print("  Finding exemplars...")
    exemplars = codebook_viz.find_exemplars(vqvae, hidden_states, config.n_exemplars)
    # Plot gallery for top-5 most used codes
    cluster_sizes = vqvae.quantizer.ema_cluster_size.cpu().numpy()
    top_codes = np.argsort(cluster_sizes)[::-1][:5]
    for code_id in top_codes:
        if len(exemplars.get(code_id, [])) > 0:
            codebook_viz.plot_exemplar_gallery(
                code_id, exemplars[code_id], hidden_states,
                transformer, source_layer, config,
                str(output_dir / f"exemplars_code_{code_id}.png"),
            )
    print(f"  Saved exemplar galleries for top codes")

    # ---------------------------------------------------------------
    # 2. Sequence analysis
    # ---------------------------------------------------------------
    print("\n=== Sequence Analysis ===")

    print("  Extracting code sequences...")
    codes = sequence_analysis.extract_code_sequences(
        hidden_states, vqvae, device=device,
    )
    print(f"  {len(codes)} codes, range [{codes.min()}, {codes.max()}]")

    zipf = sequence_analysis.zipf_analysis(codes, config.zipf_min_count)
    print(f"  Zipf alpha={zipf['alpha']:.3f}, R²={zipf['r_squared']:.3f}")
    sequence_analysis.plot_zipf(zipf, str(output_dir / "zipf.png"))

    rates = sequence_analysis.entropy_rate(codes, config.max_ngram_order)
    print(f"  Entropy rates: {[f'{r:.3f}' for r in rates]}")
    sequence_analysis.plot_entropy_rate(rates, str(output_dir / "entropy_rate.png"))

    cond_ent = sequence_analysis.conditional_entropy(codes, K)
    mi_bigram = sequence_analysis.mutual_information_bigram(codes, K)
    excess_ent = sequence_analysis.excess_entropy(codes, K)
    print(f"  Conditional entropy: {cond_ent:.3f} bits")
    print(f"  Bigram MI: {mi_bigram:.3f} bits")
    print(f"  Excess entropy: {excess_ent:.3f} bits")

    sequence_analysis.plot_mi_decay(codes, K, output_path=str(output_dir / "mi_decay.png"))
    print(f"  Saved mi_decay.png")

    trans_matrix = sequence_analysis.compute_transition_matrix(codes, K)
    sequence_analysis.plot_transition_matrix(
        trans_matrix, output_path=str(output_dir / "transition_matrix.png"),
    )
    print(f"  Saved transition_matrix.png")

    # ---------------------------------------------------------------
    # 3. Concept manipulation (sample experiment)
    # ---------------------------------------------------------------
    print("\n=== Concept Manipulation ===")

    # Use first 64 hidden states as a sample input
    sample_len = min(64, len(hidden_states))
    sample_spec = torch.randn(1, sample_len, transformer.config.n_freq).to(device)

    scan_pos = sample_len // 2
    print(f"  Concept scan at position {scan_pos}...")
    scan = concept_manipulation.concept_scan(
        transformer, vqvae, sample_spec, scan_pos, source_layer,
    )
    concept_manipulation.plot_concept_scan(
        scan, config, str(output_dir / "concept_scan.png"),
    )
    print(f"  Saved concept_scan.png")

    # ---------------------------------------------------------------
    # 4. Context analysis
    # ---------------------------------------------------------------
    print("\n=== Context Analysis ===")

    if metadata_path and Path(metadata_path).exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        grouped = context_analysis.group_codes_by_metadata(codes, metadata)
    else:
        print("  No metadata file — using single group")
        grouped = {"all": codes}

    report = context_analysis.generate_context_report(grouped, K, config)
    report_path = output_dir / "context_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved context_report.md")

    # ---------------------------------------------------------------
    # 5. Compositionality
    # ---------------------------------------------------------------
    print("\n=== Compositionality Tests ===")

    prod = compositionality.bigram_productivity(codes, K)
    print(f"  Bigram productivity: {prod['productivity_ratio']:.1%} "
          f"({prod['observed']}/{prod['possible']})")
    compositionality.plot_bigram_productivity(
        prod, str(output_dir / "bigram_productivity.png"),
    )

    pos_ind = compositionality.positional_independence(codes, hidden_states, vqvae, K)
    print(f"  Positional independence: mean |rho| = {pos_ind['mean_abs_correlation']:.3f} "
          f"({pos_ind['codes_tested']} codes tested)")
    compositionality.plot_positional_independence(
        pos_ind, str(output_dir / "positional_independence.png"),
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    summary = {
        "source_layer": source_layer,
        "codebook_size": K,
        "n_frames": len(hidden_states),
        "n_codes": len(codes),
        "zipf_alpha": zipf["alpha"],
        "zipf_r_squared": zipf["r_squared"],
        "conditional_entropy": cond_ent,
        "bigram_mi": mi_bigram,
        "excess_entropy": excess_ent,
        "entropy_rates": rates,
        "bigram_productivity": prod["productivity_ratio"],
        "positional_independence": pos_ind["mean_abs_correlation"],
    }
    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VQ-VAE analysis suite (Phase 8.4)",
    )
    parser.add_argument(
        "--transformer-checkpoint", required=True,
        help="Path to transformer .pt checkpoint",
    )
    parser.add_argument(
        "--vqvae-checkpoint", required=True,
        help="Path to VQ-VAE .pt checkpoint",
    )
    parser.add_argument(
        "--hidden-states", required=True,
        help="Path to hidden states .npy file",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for figures and reports",
    )
    parser.add_argument(
        "--source-layer", type=int, default=4,
        help="1-indexed transformer layer (default: 4)",
    )
    parser.add_argument(
        "--metadata", default=None,
        help="Path to metadata.json (optional)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for inference (default: cpu)",
    )

    args = parser.parse_args()
    run_analysis(
        transformer_path=args.transformer_checkpoint,
        vqvae_path=args.vqvae_checkpoint,
        hidden_states_path=args.hidden_states,
        output_dir=args.output_dir,
        source_layer=args.source_layer,
        metadata_path=args.metadata,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
