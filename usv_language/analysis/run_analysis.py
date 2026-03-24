"""CLI entry point for the Phase 8.4 analysis suite.

Usage:
    python -m usv_language.analysis.run_analysis \
        --transformer-checkpoint best.pt \
        --vqvae-checkpoint vqvae_best.pt \
        --hidden-states hidden_states_layer4.npy \
        --output-dir analysis_output \
        --source-layer 4

Orchestrates all six analysis modules:
1. Codebook visualization (profiles, usage, projection, exemplars)
2. Sequence analysis (Zipf, transitions, entropy, MI)
3. Concept manipulation (injection, scan)
4. Context analysis (group comparison, if metadata available)
5. Compositionality (bigram productivity, positional independence)
6. Information theory (MLE Zipf, bias-corrected entropy, idioms, burstiness)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import torch

from usv_language.analysis.config import AnalysisConfig
from usv_language.analysis.transformer_suffix import decode_hidden_to_spectrogram
from usv_language.models.transformer import SpectrogramTransformer
from usv_language.models.vqvae import HiddenStateVQVAE, VQVAEConfig
from usv_language.training.train_vqvae import infer_hidden_state_layer
from usv_language.training.train_transformer import coerce_transformer_config

matplotlib.use("Agg")


def coerce_vqvae_config(raw_config: VQVAEConfig | dict | None) -> VQVAEConfig:
    """Normalize checkpoint config payloads across older/newer formats."""
    if raw_config is None:
        return VQVAEConfig()
    if isinstance(raw_config, VQVAEConfig):
        return raw_config
    if isinstance(raw_config, dict):
        return VQVAEConfig(**raw_config)
    raise TypeError(f"Unsupported VQ-VAE config payload: {type(raw_config)!r}")


def resolve_device(device_name: str) -> torch.device:
    """Resolve a requested device with safe CPU fallback."""
    requested = torch.device(device_name)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print(f"Requested device '{device_name}' is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return requested


def _record_section_status(
    section_status: dict[str, dict[str, str]],
    section: str,
    status: str,
    reason: str | None = None,
) -> None:
    entry = {"status": status}
    if reason:
        entry["reason"] = reason
    section_status[section] = entry


def _run_section(
    section: str,
    section_status: dict[str, dict[str, str]],
    func,
):
    try:
        result = func()
    except Exception as exc:
        _record_section_status(section_status, section, "failed", str(exc))
        print(f"  Warning: {section} failed: {exc}")
        return None

    _record_section_status(section_status, section, "completed")
    return result


def _skip_section(
    section: str,
    section_status: dict[str, dict[str, str]],
    reason: str,
) -> None:
    _record_section_status(section_status, section, "skipped", reason)
    print(f"  Skipping {section}: {reason}")


def validate_vqvae_provenance(
    checkpoint: dict,
    hidden_states_path: str,
    source_layer: int,
) -> None:
    """Reject analysis runs that pair a VQ-VAE checkpoint with the wrong artifact."""
    provenance = checkpoint.get("provenance")
    if not provenance:
        return

    expected_path = provenance.get("hidden_states_path")
    if expected_path:
        provided_path = str(Path(hidden_states_path).resolve())
        expected_path = str(Path(expected_path).resolve())
        same_file = False
        try:
            same_file = os.path.samefile(provided_path, expected_path)
        except (FileNotFoundError, OSError):
            same_file = provided_path == expected_path
        if not same_file:
            raise ValueError(
                "VQ-VAE checkpoint was trained on a different hidden-state artifact: "
                f"{expected_path}"
            )

    expected_layer = provenance.get("source_layer")
    if expected_layer is not None and int(expected_layer) != source_layer:
        raise ValueError(
            "Requested source_layer does not match the VQ-VAE checkpoint provenance: "
            f"{source_layer} vs {expected_layer}"
        )


def validate_analysis_inputs(
    hidden_states_path: str,
    hidden_states: np.ndarray,
    source_layer: int,
    transformer: SpectrogramTransformer,
    metadata: dict | None = None,
) -> None:
    """Validate layer and metadata compatibility for analysis inputs."""
    if source_layer < 1 or source_layer > transformer.config.n_layers:
        raise ValueError(
            f"source_layer must be in [1, {transformer.config.n_layers}], got {source_layer}"
        )

    inferred_layer = infer_hidden_state_layer(Path(hidden_states_path))
    if inferred_layer is not None and inferred_layer != source_layer:
        raise ValueError(
            "Hidden-state filename layer does not match requested source_layer: "
            f"{inferred_layer} vs {source_layer}"
        )

    if metadata is None:
        return

    expected_frames = metadata.get("total_frames")
    if expected_frames is not None and int(expected_frames) != hidden_states.shape[0]:
        raise ValueError(
            "Metadata total_frames does not match hidden-state array length: "
            f"{expected_frames} vs {hidden_states.shape[0]}"
        )

    expected_width = metadata.get("d_model")
    if expected_width is not None and int(expected_width) != hidden_states.shape[1]:
        raise ValueError(
            "Metadata d_model does not match hidden-state width: "
            f"{expected_width} vs {hidden_states.shape[1]}"
        )


def load_analysis_metadata(
    hidden_states_path: str,
    metadata_path: str | None = None,
) -> tuple[dict | None, str | None, str | None]:
    """Load metadata explicitly or infer adjacent extractor metadata."""
    resolved_hidden_states = Path(hidden_states_path).resolve()

    if metadata_path is not None:
        resolved_metadata = Path(metadata_path).resolve()
        if not resolved_metadata.exists():
            raise FileNotFoundError(f"Metadata file not found: {resolved_metadata}")
        with open(resolved_metadata, encoding="utf-8") as f:
            return json.load(f), str(resolved_metadata), "explicit"

    adjacent_metadata = resolved_hidden_states.with_name("metadata.json")
    if adjacent_metadata.exists():
        with open(adjacent_metadata, encoding="utf-8") as f:
            return json.load(f), str(adjacent_metadata), "adjacent"

    return None, None, None


def load_models(
    transformer_path: str,
    vqvae_path: str,
    device: torch.device,
) -> tuple[SpectrogramTransformer, HiddenStateVQVAE, dict]:
    """Load frozen transformer and VQ-VAE from checkpoints."""
    t_ckpt = torch.load(transformer_path, map_location=device, weights_only=False)
    t_config = coerce_transformer_config(t_ckpt.get("config"))
    transformer = SpectrogramTransformer(t_config).to(device)
    transformer.load_state_dict(t_ckpt["model_state_dict"])
    transformer.eval()

    v_ckpt = torch.load(vqvae_path, map_location=device, weights_only=False)
    v_config = coerce_vqvae_config(v_ckpt.get("config"))
    vqvae = HiddenStateVQVAE(v_config).to(device)
    vqvae.load_state_dict(v_ckpt["model_state_dict"])
    vqvae.eval()

    return transformer, vqvae, v_ckpt


def build_analysis_sample_spectrogram(
    hidden_states: np.ndarray,
    transformer: SpectrogramTransformer,
    source_layer: int,
    device: torch.device,
    sample_len: int = 64,
) -> torch.Tensor:
    """Decode a real hidden-state slice into spectrogram space for analysis."""
    if hidden_states.ndim != 2:
        raise ValueError(
            f"hidden_states must have shape (N, d_model), got {hidden_states.shape}"
        )
    if len(hidden_states) == 0:
        raise ValueError("hidden_states must contain at least one frame")

    sample_len = min(sample_len, len(hidden_states))
    hidden_slice = torch.from_numpy(hidden_states[:sample_len]).float().unsqueeze(0).to(device)
    with torch.no_grad():
        return decode_hidden_to_spectrogram(transformer, hidden_slice, source_layer)


def run_analysis(
    transformer_path: str,
    vqvae_path: str,
    hidden_states_path: str,
    output_dir: str,
    source_layer: int = 4,
    metadata_path: str | None = None,
    device_name: str = "cpu",
) -> None:
    """Run the full analysis pipeline."""
    from usv_language.analysis import codebook_viz
    from usv_language.analysis import compositionality
    from usv_language.analysis import concept_manipulation
    from usv_language.analysis import context_analysis
    from usv_language.analysis import information_theory
    from usv_language.analysis import sequence_analysis

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)

    transformer_path = str(Path(transformer_path))
    vqvae_path = str(Path(vqvae_path))
    hidden_states_path = str(Path(hidden_states_path))
    for required_path in (transformer_path, vqvae_path, hidden_states_path):
        if not Path(required_path).exists():
            raise FileNotFoundError(f"Required analysis artifact not found: {required_path}")

    config = AnalysisConfig(source_layer=source_layer)

    print("Loading models...")
    transformer, vqvae, vqvae_checkpoint = load_models(transformer_path, vqvae_path, device)
    K = vqvae.config.codebook_size
    print(f"  Transformer: {transformer.count_parameters():,} params")
    print(f"  VQ-VAE: {vqvae.count_parameters():,} params, K={K}")
    validate_vqvae_provenance(vqvae_checkpoint, hidden_states_path, source_layer)

    print(f"Loading hidden states from {hidden_states_path}...")
    hidden_states = np.load(hidden_states_path)
    print(f"  Shape: {hidden_states.shape}")
    if hidden_states.ndim != 2:
        raise ValueError(
            f"Hidden states must have shape (N, d_model), got {hidden_states.shape}"
        )
    if hidden_states.shape[0] == 0:
        raise ValueError("Hidden states array is empty; analysis requires at least one frame.")
    if hidden_states.shape[1] != transformer.config.d_model:
        raise ValueError(
            "Hidden state width does not match transformer hidden size: "
            f"{hidden_states.shape[1]} vs {transformer.config.d_model}."
        )
    if transformer.config.d_model != vqvae.config.d_model:
        raise ValueError(
            "Transformer/VQ-VAE hidden sizes are incompatible: "
            f"{transformer.config.d_model} vs {vqvae.config.d_model}."
        )

    metadata, resolved_metadata_path, metadata_source = load_analysis_metadata(
        hidden_states_path,
        metadata_path=metadata_path,
    )
    if metadata is not None:
        layers_extracted = metadata.get("layers_extracted")
        if layers_extracted and source_layer not in layers_extracted:
            raise ValueError(
                f"Requested source layer {source_layer} is not present in metadata "
                f"layers_extracted={layers_extracted}."
            )
    validate_analysis_inputs(
        hidden_states_path, hidden_states, source_layer, transformer, metadata,
    )

    section_status: dict[str, dict[str, str]] = {}

    print("\n=== Codebook Visualization ===")

    def _codebook_viz_section() -> None:
        print("  Decoding all codebook entries...")
        profiles = codebook_viz.decode_all_entries(transformer, vqvae, source_layer)
        codebook_viz.plot_decoded_profiles(
            profiles, config, str(output_dir_path / "codebook_profiles.png"),
        )
        print("  Saved codebook_profiles.png")

        codebook_viz.plot_codebook_usage(vqvae, str(output_dir_path / "codebook_usage.png"))
        print("  Saved codebook_usage.png")

        codebook_viz.plot_codebook_projection(
            vqvae, profiles, config, str(output_dir_path / "codebook_projection.png"),
        )
        print("  Saved codebook_projection.png")

        print("  Finding exemplars...")
        exemplars = codebook_viz.find_exemplars(vqvae, hidden_states, config.n_exemplars)
        cluster_sizes = vqvae.quantizer.ema_cluster_size.cpu().numpy()
        top_codes = np.argsort(cluster_sizes)[::-1][:5]
        for code_id in top_codes:
            if len(exemplars.get(code_id, [])) > 0:
                codebook_viz.plot_exemplar_gallery(
                    code_id,
                    exemplars[code_id],
                    hidden_states,
                    transformer,
                    source_layer,
                    config,
                    str(output_dir_path / f"exemplars_code_{code_id}.png"),
                )
        print("  Saved exemplar galleries for top codes")

    _run_section("codebook_visualization", section_status, _codebook_viz_section)

    print("\n=== Sequence Analysis ===")
    codes: np.ndarray | None = None
    zipf: dict | None = None
    rates: list[float] = []
    cond_ent: float | None = None
    mi_bigram: float | None = None
    excess_ent: float | None = None

    def _sequence_analysis_section():
        print("  Extracting code sequences...")
        codes_local = sequence_analysis.extract_code_sequences(
            hidden_states, vqvae, device=device,
        )
        print(f"  {len(codes_local)} codes, range [{codes_local.min()}, {codes_local.max()}]")

        zipf_local = sequence_analysis.zipf_analysis(codes_local, config.zipf_min_count)
        print(f"  Zipf alpha={zipf_local['alpha']:.3f}, R²={zipf_local['r_squared']:.3f}")
        sequence_analysis.plot_zipf(zipf_local, str(output_dir_path / "zipf.png"))

        rates_local = sequence_analysis.entropy_rate(codes_local, config.max_ngram_order)
        print(f"  Entropy rates: {[f'{r:.3f}' for r in rates_local]}")
        sequence_analysis.plot_entropy_rate(rates_local, str(output_dir_path / "entropy_rate.png"))

        cond_ent_local = sequence_analysis.conditional_entropy(codes_local, K)
        mi_bigram_local = sequence_analysis.mutual_information_bigram(codes_local, K)
        excess_ent_local = sequence_analysis.excess_entropy(codes_local, K)
        print(f"  Conditional entropy: {cond_ent_local:.3f} bits")
        print(f"  Bigram MI: {mi_bigram_local:.3f} bits")
        print(f"  Excess entropy: {excess_ent_local:.3f} bits")

        sequence_analysis.plot_mi_decay(
            codes_local, K, output_path=str(output_dir_path / "mi_decay.png"),
        )
        print("  Saved mi_decay.png")

        trans_matrix = sequence_analysis.compute_transition_matrix(codes_local, K)
        sequence_analysis.plot_transition_matrix(
            trans_matrix, output_path=str(output_dir_path / "transition_matrix.png"),
        )
        print("  Saved transition_matrix.png")

        return (
            codes_local,
            zipf_local,
            rates_local,
            cond_ent_local,
            mi_bigram_local,
            excess_ent_local,
        )

    sequence_result = _run_section("sequence_analysis", section_status, _sequence_analysis_section)
    if sequence_result is not None:
        codes, zipf, rates, cond_ent, mi_bigram, excess_ent = sequence_result

    print("\n=== Concept Manipulation ===")

    def _concept_manipulation_section() -> None:
        sample_spec = build_analysis_sample_spectrogram(
            hidden_states, transformer, source_layer, device, sample_len=64,
        )
        scan_pos = sample_spec.shape[1] // 2
        print(f"  Concept scan at position {scan_pos}...")
        scan = concept_manipulation.concept_scan(
            transformer, vqvae, sample_spec, scan_pos, source_layer,
        )
        concept_manipulation.plot_concept_scan(
            scan, config, str(output_dir_path / "concept_scan.png"),
        )
        print("  Saved concept_scan.png")

    _run_section("concept_manipulation", section_status, _concept_manipulation_section)

    print("\n=== Context Analysis ===")
    if codes is None:
        _skip_section(
            "context_analysis",
            section_status,
            "sequence analysis did not produce code sequences",
        )
    else:
        def _context_analysis_section() -> None:
            if metadata is not None:
                grouped = context_analysis.group_codes_by_metadata(codes, metadata)
            else:
                print("  No metadata file; using single group")
                grouped = {"all": codes}

            report = context_analysis.generate_context_report(grouped, K, config)
            report_path = output_dir_path / "context_report.md"
            report_path.write_text(report, encoding="utf-8")
            print("  Saved context_report.md")

        _run_section("context_analysis", section_status, _context_analysis_section)

    print("\n=== Compositionality Tests ===")
    prod: dict | None = None
    pos_ind: dict | None = None
    if codes is None:
        _skip_section(
            "compositionality",
            section_status,
            "sequence analysis did not produce code sequences",
        )
    else:
        def _compositionality_section():
            prod_local = compositionality.bigram_productivity(codes, K)
            print(
                "  Bigram productivity: "
                f"{prod_local['productivity_ratio']:.1%} "
                f"({prod_local['observed']}/{prod_local['possible']})"
            )
            compositionality.plot_bigram_productivity(
                prod_local, str(output_dir_path / "bigram_productivity.png"),
            )

            pos_ind_local = compositionality.positional_independence(
                codes, hidden_states, vqvae, K,
            )
            print(
                "  Positional independence: mean |rho| = "
                f"{pos_ind_local['mean_abs_correlation']:.3f} "
                f"({pos_ind_local['codes_tested']} codes tested)"
            )
            compositionality.plot_positional_independence(
                pos_ind_local, str(output_dir_path / "positional_independence.png"),
            )
            return prod_local, pos_ind_local

        compositionality_result = _run_section(
            "compositionality", section_status, _compositionality_section,
        )
        if compositionality_result is not None:
            prod, pos_ind = compositionality_result

    print("\n=== Information Theory ===")
    zipf_mle = None
    zipf_ent = None
    ent_rate = None
    idioms = None
    prod_ci = None
    burst = None
    if codes is None:
        _skip_section(
            "information_theory",
            section_status,
            "sequence analysis did not produce code sequences",
        )
    else:
        def _information_theory_section():
            zipf_mle_local = information_theory.zipf_exponent_mle(codes)
            print(
                f"  Zipf MLE: count_alpha={zipf_mle_local.alpha:.3f}, "
                f"rank_alpha={zipf_mle_local.rank_alpha:.3f}, "
                f"xmin={zipf_mle_local.xmin:.1f}, p={zipf_mle_local.p_value:.3f}"
            )

            zipf_ent_local = information_theory.zipf_via_shannon_entropy(codes, K)
            print(
                "  Zipf entropy inversion: "
                f"rank_alpha={zipf_ent_local.alpha_estimate:.3f}, "
                f"H_obs={zipf_ent_local.entropy_observed:.3f} bits, "
                f"method={zipf_ent_local.method}"
            )

            ent_rate_local = information_theory.entropy_rate(
                codes, max_order=config.max_ngram_order,
            )
            print(
                "  Entropy rate (corrected): "
                f"{[f'{r:.3f}' for r in ent_rate_local.rates_corrected]}"
            )
            if ent_rate_local.convergence_order > 0:
                print(f"  Convergence at order {ent_rate_local.convergence_order}")

            idioms_local = information_theory.ngram_idioms(codes, K, max_n=4, n_shuffles=100)
            print(f"  Significant idioms: {len(idioms_local)}")
            for idiom in idioms_local[:5]:
                print(f"    {idiom.ngram}: z={idiom.z_score:.2f}, count={idiom.observed_count}")

            prod_ci_local = information_theory.ngram_productivity(
                codes, K, n=2, n_bootstrap=500,
            )
            print(
                "  Bigram productivity (bootstrap): "
                f"{prod_ci_local.productivity_ratio:.1%} "
                f"null CI [{prod_ci_local.null_ci_lower:.1%}, {prod_ci_local.null_ci_upper:.1%}]"
            )

            cond_ent_by_lag_local = information_theory.conditional_entropy_by_lag(
                codes, K, max_lag=10,
            )
            print(f"  Cond. entropy by lag (1-3): {[f'{v:.3f}' for v in cond_ent_by_lag_local[:3]]}")

            mi_decay_local = information_theory.mutual_information_rate(codes, K, max_lag=20)
            print(f"  MI decay (lags 1-3): {[f'{v:.3f}' for v in mi_decay_local[:3]]}")

            hop_duration_s = 128 / 300_000
            change_mask = np.concatenate([[True], np.diff(codes) != 0])
            change_times = np.where(change_mask)[0].astype(np.float64) * hop_duration_s
            burst_local = information_theory.burstiness_coefficient(change_times)
            print(
                f"  Burstiness: CV={burst_local.cv:.3f} "
                f"({burst_local.interpretation}), n_bursts={burst_local.n_bursts}"
            )
            return zipf_mle_local, zipf_ent_local, ent_rate_local, idioms_local, prod_ci_local, burst_local

        info_result = _run_section("information_theory", section_status, _information_theory_section)
        if info_result is not None:
            zipf_mle, zipf_ent, ent_rate, idioms, prod_ci, burst = info_result

    summary = {
        "source_layer": source_layer,
        "codebook_size": K,
        "n_frames": len(hidden_states),
        "n_codes": int(len(codes)) if codes is not None else 0,
        "zipf_alpha": zipf["alpha"] if zipf is not None else None,
        "zipf_r_squared": zipf["r_squared"] if zipf is not None else None,
        "conditional_entropy": cond_ent,
        "bigram_mi": mi_bigram,
        "excess_entropy": excess_ent,
        "entropy_rates": rates,
        "bigram_productivity": prod["productivity_ratio"] if prod is not None else None,
        "positional_independence": (
            pos_ind["mean_abs_correlation"] if pos_ind is not None else None
        ),
        "zipf_mle_count_alpha": zipf_mle.alpha if zipf_mle is not None else None,
        "zipf_mle_rank_alpha": zipf_mle.rank_alpha if zipf_mle is not None else None,
        "zipf_entropy_rank_alpha": (
            zipf_ent.alpha_estimate if zipf_ent is not None else None
        ),
        "entropy_rates_corrected": ent_rate.rates_corrected if ent_rate is not None else [],
        "entropy_convergence_order": ent_rate.convergence_order if ent_rate is not None else 0,
        "n_significant_idioms": len(idioms) if idioms is not None else 0,
        "bigram_productivity_null_ci": (
            [prod_ci.null_ci_lower, prod_ci.null_ci_upper] if prod_ci is not None else None
        ),
        "burstiness_cv": burst.cv if burst is not None else None,
        "burstiness_interpretation": burst.interpretation if burst is not None else None,
        "artifacts": {
            "transformer_checkpoint": str(Path(transformer_path).resolve()),
            "vqvae_checkpoint": str(Path(vqvae_path).resolve()),
            "hidden_states": str(Path(hidden_states_path).resolve()),
            "metadata": resolved_metadata_path,
            "metadata_source": metadata_source,
            "vqvae_provenance": vqvae_checkpoint.get("provenance"),
        },
        "section_status": section_status,
    }
    summary_path = output_dir_path / "analysis_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
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
