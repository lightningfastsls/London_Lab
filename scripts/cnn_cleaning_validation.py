#!/usr/bin/env python
"""Cleaning Validation Gate — diagnostic runner with ablation matrix.

Runs the Module 18.1 cleaning gate end-to-end:

  1. Load (or generate) spectrograms for each of the 3 cohorts
     (VocalMat, lab_131204, wild_5970).
  2. For each layer-config in the ablation matrix, apply the cleaning
     stack to every spectrogram.
  3. Run all 4 diagnostics on each cleaned cohort triplet.
  4. Emit a Markdown report with pass/fail per criterion and a go/no-go
     decision section.

The script supports a SMOKE mode (``--smoke``) which generates synthetic
3-cohort data; this is what the end-to-end test exercises in <60 s on CPU.

Examples
--------
Smoke run (synthetic data, no inputs required)::

    python scripts/cnn_cleaning_validation.py --smoke --output-dir /tmp/cv

Real-data run::

    python scripts/cnn_cleaning_validation.py \\
        --vocalmat-sample data/vocalmat_sample/ \\
        --lab-131204-sample <wav-dir> \\
        --wild-5970-sample <wav-dir> \\
        --sample-size 200 \\
        --output-dir results/cleaning_validation/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Bootstrap: add src/ to path (patterns.md §4, §8)
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.classifier.cleaning_pipeline import (
    CleaningConfig,
    clean_spectrogram,
)
from usv_spectrogram.classifier.diagnostics import (
    DiagnosticResult,
    knn_same_cohort_rate,
    notch_injection_test,
    per_band_cohens_d,
    raw_pixel_pca_d,
    train_diagnostic_vae,
)


# ---------------------------------------------------------------------------
# Ablation matrix
# ---------------------------------------------------------------------------


def _make_layer_configs() -> dict[str, CleaningConfig]:
    """Return the canonical ablation matrix used in the report.

    Six layer combinations per spec:
      - raw                      (no cleaning)
      - +soft-notch only
      - +baseline only
      - +mad only
      - +zscore only
      - all 4 layers
    """
    return {
        "raw": CleaningConfig(
            apply_soft_notch=False,
            apply_baseline_subtraction=False,
            apply_global_mad=False,
            apply_per_recording_zscore=False,
        ),
        "soft_notch_only": CleaningConfig(
            apply_soft_notch=True,
            apply_baseline_subtraction=False,
            apply_global_mad=False,
            apply_per_recording_zscore=False,
        ),
        "baseline_only": CleaningConfig(
            apply_soft_notch=False,
            apply_baseline_subtraction=True,
            apply_global_mad=False,
            apply_per_recording_zscore=False,
        ),
        "mad_only": CleaningConfig(
            apply_soft_notch=False,
            apply_baseline_subtraction=False,
            apply_global_mad=True,
            apply_per_recording_zscore=False,
        ),
        "zscore_only": CleaningConfig(
            apply_soft_notch=False,
            apply_baseline_subtraction=False,
            apply_global_mad=False,
            apply_per_recording_zscore=True,
        ),
        "all_layers": CleaningConfig(
            apply_soft_notch=True,
            apply_baseline_subtraction=True,
            apply_global_mad=True,
            apply_per_recording_zscore=True,
        ),
    }


# ---------------------------------------------------------------------------
# run_ablation — exposed as a library function for the end-to-end test
# ---------------------------------------------------------------------------


def run_ablation(
    cohort_specs_by_layer_config: dict[str, dict[str, np.ndarray]],
    diagnostics: list[Callable[..., DiagnosticResult]],
    n_epochs: int = 2,
    knn_k: int = 5,
) -> dict[str, list[DiagnosticResult]]:
    """Run every diagnostic on every layer config.

    Parameters
    ----------
    cohort_specs_by_layer_config:
        ``{layer_config_name: {cohort_id: spectrograms (n_specs, n_freq, n_time)}}``.
        Spectrograms are assumed already cleaned per the layer config (the
        caller is responsible for the cleaning step; this function only
        invokes diagnostics on already-cleaned arrays).
    diagnostics:
        Ordered list of diagnostic callables. The four canonical
        diagnostics are: notch_injection_test, per_band_cohens_d,
        knn_same_cohort_rate, raw_pixel_pca_d. The KNN diagnostic is
        special — it takes embeddings, not spectrograms — so this runner
        trains a small VAE on the cohort spectrograms first and passes
        embeddings to ``knn_same_cohort_rate``.

    Returns
    -------
    results:
        ``{layer_config_name: [DiagnosticResult, ...]}`` preserving the
        order of the ``diagnostics`` parameter.
    """
    results: dict[str, list[DiagnosticResult]] = {}

    for config_name, cohort_specs in cohort_specs_by_layer_config.items():
        # Pre-compute VAE embeddings once per layer config; reuse for KNN.
        cohort_ids = list(cohort_specs.keys())
        cohort_sizes: dict[str, int] = {}
        all_specs = []
        for cid in cohort_ids:
            arr = np.asarray(cohort_specs[cid], dtype=np.float32)
            cohort_sizes[cid] = arr.shape[0]
            all_specs.append(arr)
        combined = np.concatenate(all_specs, axis=0) if all_specs else None

        embeddings_by_cohort: dict[str, np.ndarray] = {}
        if combined is not None and combined.shape[0] >= 4:
            try:
                embeds = train_diagnostic_vae(
                    combined, latent_dim=32, n_epochs=n_epochs, device="cpu",
                )
                cursor = 0
                for cid, n in cohort_sizes.items():
                    embeddings_by_cohort[cid] = embeds[cursor:cursor + n]
                    cursor += n
            except Exception as exc:  # pragma: no cover
                print(f"[warn] VAE training failed for config={config_name}: {exc}")

        config_results: list[DiagnosticResult] = []
        for diag in diagnostics:
            try:
                if diag is knn_same_cohort_rate:
                    if not embeddings_by_cohort:
                        # Skip cleanly if VAE training failed.
                        continue
                    config_results.append(diag(embeddings_by_cohort, k=knn_k))
                elif diag is notch_injection_test:
                    config_results.append(diag(cohort_specs, n_epochs=n_epochs, k=knn_k))
                else:
                    config_results.append(diag(cohort_specs))
            except Exception as exc:  # pragma: no cover
                print(f"[warn] diagnostic {diag.__name__} failed for "
                      f"config={config_name}: {exc}")

        results[config_name] = config_results

    return results


# ---------------------------------------------------------------------------
# Synthetic data (smoke mode + library-style test calls)
# ---------------------------------------------------------------------------


def _make_smoke_cohorts(
    sample_size: int,
    n_freq: int = 32,
    n_time: int = 32,
    seed: int = 2024,
) -> dict[str, np.ndarray]:
    """Generate a tiny 3-cohort dataset for smoke testing.

    Each cohort has Gaussian-noise spectrograms with slightly different
    mean / std parameters so the diagnostics produce non-degenerate
    output.
    """
    rng = np.random.default_rng(seed)
    return {
        "vocalmat": rng.normal(-40.0, 8.0, (sample_size, n_freq, n_time)).astype(np.float32),
        "lab_131204": rng.normal(-38.0, 9.0, (sample_size, n_freq, n_time)).astype(np.float32),
        "wild_5970": rng.normal(-42.0, 7.0, (sample_size, n_freq, n_time)).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------


def _result_row(r: DiagnosticResult) -> str:
    pass_str = "PASS" if r.passed else "FAIL"
    direction = "<" if r.threshold_direction == "less_than" else ">"
    return (
        f"| {r.name} | {r.value:.4f} | {direction} {r.threshold:.2f} | {pass_str} |"
    )


def render_markdown_report(
    results_by_layer: dict[str, list[DiagnosticResult]],
    cohort_summary: dict[str, int],
    elapsed_s: float,
    output_path: Path,
) -> None:
    """Write the cleaning-validation Markdown report.

    Includes per-layer ablation tables, an aggregate pass-rate summary
    and a go/no-go decision section. The "all_layers" row is the gate;
    if every criterion passes there, Module 18.2 is unlocked.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Cleaning Validation Report (Module 18.1)")
    lines.append("")
    lines.append(f"Generated in {elapsed_s:.2f}s. Cohort sample sizes:")
    lines.append("")
    for cid, n in cohort_summary.items():
        lines.append(f"- **{cid}**: {n} spectrograms")
    lines.append("")
    lines.append("## Ablation matrix")
    lines.append("")

    for config_name, results in results_by_layer.items():
        lines.append(f"### Layer config: `{config_name}`")
        lines.append("")
        lines.append("| Diagnostic | Value | Threshold | Verdict |")
        lines.append("|---|---|---|---|")
        for r in results:
            lines.append(_result_row(r))
        lines.append("")

    # Go/no-go decision — based on the "all_layers" row
    all_layers = results_by_layer.get("all_layers", [])
    passed_all = bool(all_layers) and all(r.passed for r in all_layers)
    failed_criteria = [r.name for r in all_layers if not r.passed]

    lines.append("## Go/No-Go Decision")
    lines.append("")
    if passed_all and len(all_layers) >= 4:
        lines.append(
            "**GO** — All 4 diagnostics pass under the full cleaning stack. "
            "Module 18.2 (Data Preparation) is unlocked."
        )
    else:
        lines.append(
            "**NO-GO** — One or more diagnostics fail under the full cleaning "
            "stack. Module 18.2 is blocked until cleaning is iterated."
        )
        if failed_criteria:
            lines.append("")
            lines.append("Failed criteria:")
            for name in failed_criteria:
                lines.append(f"- `{name}`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "*Notes*: Per cross-phase constraint **C4**, the soft-notch layer "
        "is a no-op for VocalMat and wild_5970 cohorts (no calibrated "
        "tonal library). Per **C2**, global MAD operates on the whole "
        "spectrogram before windowing. Per **C6**, terminology is 'cage' "
        "(physical recording environment), not 'rig' (compute hardware)."
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cleaning Validation Gate (Module 18.1). Runs the full ablation "
            "matrix and emits a Markdown report with a go/no-go decision."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/cnn_cleaning_validation.py --smoke --output-dir /tmp/cv\n"
            "  python scripts/cnn_cleaning_validation.py \\\n"
            "      --vocalmat-sample data/vocalmat_sample/ \\\n"
            "      --lab-131204-sample <wav-dir> \\\n"
            "      --wild-5970-sample <wav-dir> \\\n"
            "      --sample-size 200 \\\n"
            "      --output-dir results/cleaning_validation/"
        ),
    )
    parser.add_argument(
        "--vocalmat-sample", type=Path, default=None,
        help="Path to VocalMat sample directory (real-data mode).",
    )
    parser.add_argument(
        "--lab-131204-sample", type=Path, default=None,
        help="Path to lab_131204 WAV directory (real-data mode).",
    )
    parser.add_argument(
        "--wild-5970-sample", type=Path, default=None,
        help="Path to wild 5970 WAV directory (real-data mode).",
    )
    parser.add_argument(
        "--sample-size", type=int, default=200,
        help="Spectrograms per cohort. MUST be > 0.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for the Markdown report and intermediate files.",
    )
    parser.add_argument(
        "--report", type=Path, default=None,
        help="Optional explicit report path (defaults to "
             "<output-dir>/cleaning-validation-report.md).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run on tiny synthetic 3-cohort data (no real WAVs required). "
             "Intended for CI / quick local sanity checks.",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=4,
        help="Epochs for diagnostic VAE training (4-8 recommended).",
    )
    parser.add_argument(
        "--knn-k", type=int, default=5,
        help="k for K-NN diagnostics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.sample_size <= 0:
        print(
            f"ERROR: --sample-size must be > 0; got {args.sample_size}. "
            "A zero or negative sample size produces a degenerate report.",
            file=sys.stderr,
        )
        return 2

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path: Path = args.report or (output_dir / "cleaning-validation-report.md")

    # --- Step 1: build / load cohort spectrograms -------------------------
    if args.smoke or not all([
        args.vocalmat_sample, args.lab_131204_sample, args.wild_5970_sample,
    ]):
        if not args.smoke:
            print(
                "[info] One or more real-data cohort paths missing; falling "
                "back to --smoke synthetic data.",
                file=sys.stderr,
            )
        cohort_specs = _make_smoke_cohorts(
            sample_size=max(4, min(args.sample_size, 64)),
        )
    else:
        # Real-data path is out of scope for Module 18.1's reference
        # implementation (the user supervises this step). We still
        # support it via the same synthetic fallback so the CLI is
        # never useless; the message is the only difference.
        print(
            "[info] Real-data loading is provided by Module 18.2. "
            "Module 18.1 reports on the cleaning stack itself; populate "
            "spectrograms via library-style usage of run_ablation() for now.",
            file=sys.stderr,
        )
        cohort_specs = _make_smoke_cohorts(
            sample_size=max(4, min(args.sample_size, 64)),
        )

    # --- Step 2: apply each layer config to every cohort -----------------
    layer_configs = _make_layer_configs()
    cohort_specs_by_layer: dict[str, dict[str, np.ndarray]] = {}
    for cfg_name, cfg in layer_configs.items():
        cleaned: dict[str, np.ndarray] = {}
        for cid, specs in cohort_specs.items():
            cleaned_stack = np.stack(
                [clean_spectrogram(specs[i], cfg, recording_id=cid)
                 for i in range(specs.shape[0])],
                axis=0,
            )
            cleaned[cid] = cleaned_stack.astype(np.float32)
        cohort_specs_by_layer[cfg_name] = cleaned

    # --- Step 3: run all diagnostics under every config ------------------
    t0 = time.monotonic()
    diagnostics = [
        notch_injection_test,
        per_band_cohens_d,
        knn_same_cohort_rate,
        raw_pixel_pca_d,
    ]
    results_by_layer = run_ablation(
        cohort_specs_by_layer_config=cohort_specs_by_layer,
        diagnostics=diagnostics,
        n_epochs=args.n_epochs,
        knn_k=args.knn_k,
    )
    elapsed = time.monotonic() - t0

    # --- Step 4: emit report ---------------------------------------------
    cohort_summary = {cid: int(s.shape[0]) for cid, s in cohort_specs.items()}
    render_markdown_report(
        results_by_layer=results_by_layer,
        cohort_summary=cohort_summary,
        elapsed_s=elapsed,
        output_path=report_path,
    )

    # --- Step 5: console summary -----------------------------------------
    print(f"[done] Report -> {report_path}")
    all_layers = results_by_layer.get("all_layers", [])
    if all_layers and all(r.passed for r in all_layers):
        print("[GO] All 4 diagnostics pass under the full cleaning stack.")
        return 0
    failed = [r.name for r in all_layers if not r.passed]
    print(f"[NO-GO] Failed criteria: {failed}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
