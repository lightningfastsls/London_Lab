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

from usv_spectrogram.classifier import TARGET_SAMPLE_RATE_HZ
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
from usv_spectrogram.corpus import STFT_HOP, STFT_N_FFT


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
# Real-data loader (Module 18.2a) — VocalMat PNGs + lab/wild WAV STFTs
# ---------------------------------------------------------------------------

# Common target shape across the 3 cohorts. VocalMat ships 227×227 RGB
# PNGs (AlexNet input convention from their CNN pipeline); we resize the
# WAV-derived STFTs to match so K-NN / PCA / VAE can compare apples-to-
# apples. The per_band_cohens_d diagnostic re-bins by Hz inside each
# cohort, so per-bin frequency alignment is not required, but shape
# alignment IS — concat/stack steps elsewhere assume uniform shape.
# 227 is a VocalMat-pipeline convention, NOT a corpus invariant.
_REAL_TARGET_SHAPE: tuple[int, int] = (227, 227)

# Lab/wild STFT n_fft / hop are imported from `corpus.STFT_N_FFT` and
# `corpus.STFT_HOP` (canonical, ADR-002) — do NOT redeclare them here.
# At 250 kHz target SR, n_fft=512 gives 257 frequency bins covering
# 0..125 kHz; the bilinear resize to 227 is a mild downsample.
# Target sample rate is `classifier.TARGET_SAMPLE_RATE_HZ` (250 kHz,
# VocalMat-aligned, NOT corpus.SAMPLE_RATE_HZ which is 300 kHz for
# the detection pipeline — see CleaningConfig cross-phase constraint C1).
#
# Window duration per sampled spectrogram, in seconds. 0.22s matches
# Module 18.2b's planned patch size (ROADMAP D1) so the gate's
# diagnostic windows live in the same time scale as eventual training
# patches. This is a classifier-pipeline analysis parameter, not a
# corpus invariant.
_REAL_WINDOW_DURATION_S: float = 0.22


def _resize_2d(spec: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2-D array to ``target_shape`` using bilinear interp.

    Implemented with two passes of ``np.interp`` (one per axis) to avoid
    a heavy scipy.ndimage dependency. For modest size changes (257→227
    on freq, ~430→227 on time at 0.22s windows) the result is visually
    indistinguishable from scipy.ndimage.zoom.
    """
    h, w = spec.shape
    th, tw = target_shape
    if (h, w) == (th, tw):
        return spec.astype(np.float32, copy=False)
    # Resample along axis 1 (time) first.
    src_t = np.linspace(0.0, 1.0, w, dtype=np.float64)
    dst_t = np.linspace(0.0, 1.0, tw, dtype=np.float64)
    along_time = np.empty((h, tw), dtype=np.float32)
    for i in range(h):
        along_time[i] = np.interp(dst_t, src_t, spec[i]).astype(np.float32)
    # Then resample along axis 0 (frequency).
    src_f = np.linspace(0.0, 1.0, h, dtype=np.float64)
    dst_f = np.linspace(0.0, 1.0, th, dtype=np.float64)
    out = np.empty((th, tw), dtype=np.float32)
    for j in range(tw):
        out[:, j] = np.interp(dst_f, src_f, along_time[:, j]).astype(np.float32)
    return out


def _png_to_luminance(png_path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    """Load a VocalMat PNG, convert RGB→luminance, resize to target_shape.

    VocalMat PNGs are 227×227 RGB renderings (likely a perceptual
    colormap such as parula). For the cleaning gate's pixel-distribution
    diagnostics we just need a scalar per pixel — PIL's ``L`` conversion
    applies the standard 0.299/0.587/0.114 luminance weights, which is
    a defensible scalar surrogate for "intensity". Returned array is
    float32, range [0, 1] (divided by 255).
    """
    from PIL import Image

    with Image.open(png_path) as im:
        gray = np.array(im.convert("L"), dtype=np.float32) / 255.0
    if gray.shape != target_shape:
        gray = _resize_2d(gray, target_shape)
    return gray


def _wav_to_spectrograms(
    wav_paths: list[Path],
    n_windows: int,
    target_sr: int,
    target_shape: tuple[int, int],
    window_seconds: float,
    n_fft: int,
    hop: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute ``n_windows`` STFT spectrograms sampled across ``wav_paths``.

    For each window: pick a random WAV, pick a random start sample, slice
    a ``window_seconds``-long chunk, resample 300 → target_sr if needed,
    STFT, convert magnitude to dB, resize to ``target_shape``. Return a
    stack of shape ``(n_windows, *target_shape)`` float32.

    The lab/wild source recordings on disk are 300 kHz (corpus canonical).
    We resample with ``scipy.signal.resample_poly`` (rational 5/6) to
    250 kHz before STFT so the freq axis spans 0..125 kHz, matching
    VocalMat's 250 kHz Nyquist.
    """
    import soundfile as sf
    from scipy import signal as scisig

    if not wav_paths:
        raise RuntimeError("_wav_to_spectrograms: no WAV files supplied")

    n_window_samples_target = int(round(window_seconds * target_sr))
    # We resample after slicing, so the source-domain sample count is
    # window_seconds * source_sr. Source sr known per file.
    out = np.empty((n_windows, *target_shape), dtype=np.float32)
    failures = 0
    i = 0
    attempts = 0
    max_attempts = n_windows * 5  # guard against pathological short WAVs

    while i < n_windows and attempts < max_attempts:
        attempts += 1
        wav_path = wav_paths[int(rng.integers(len(wav_paths)))]
        try:
            samples, src_sr = sf.read(str(wav_path), dtype="float32",
                                       always_2d=False)
        except Exception:
            failures += 1
            continue
        if samples.ndim > 1:
            samples = samples[:, 0]  # take first channel
        n_src_window = int(round(window_seconds * src_sr))
        if samples.size < n_src_window:
            failures += 1
            continue
        start = int(rng.integers(samples.size - n_src_window + 1))
        chunk = samples[start:start + n_src_window]

        # Resample 300 → target_sr (skip if already at target).
        if src_sr != target_sr:
            from math import gcd
            g = gcd(int(src_sr), int(target_sr))
            up = target_sr // g
            down = src_sr // g
            chunk = scisig.resample_poly(chunk, up=up, down=down).astype(np.float32)

        if chunk.size < n_fft:
            failures += 1
            continue

        # STFT → magnitude → dB.
        f, t, Z = scisig.stft(
            chunk, fs=target_sr, nperseg=n_fft, noverlap=n_fft - hop,
            boundary=None, padded=False,
        )
        mag = np.abs(Z).astype(np.float32)
        # Avoid log(0). The 1e-10 floor is consistent with the
        # cleaning pipeline's eps elsewhere (we don't import the
        # exact constant to avoid coupling to classifier internals).
        spec_db = 20.0 * np.log10(mag + 1e-10).astype(np.float32)

        out[i] = _resize_2d(spec_db, target_shape)
        i += 1

    if i < n_windows:
        raise RuntimeError(
            f"Failed to produce {n_windows} spectrograms after "
            f"{attempts} attempts ({failures} WAV read/length failures). "
            "Check that the WAV directory contains usable recordings."
        )
    return out


def _load_real_cohorts(
    vocalmat_dir: Path,
    lab_wav_dir: Path,
    wild_wav_dir: Path,
    sample_size: int,
    seed: int = 1729,
    target_shape: tuple[int, int] = _REAL_TARGET_SHAPE,
    target_sample_rate_hz: int = TARGET_SAMPLE_RATE_HZ,
) -> dict[str, np.ndarray]:
    """Load real-data 3-cohort spectrogram dict for the gate.

    Returned dict:

    - ``vocalmat``: ``(sample_size, *target_shape)`` from random PNGs in
      ``vocalmat_dir/*/*.png`` (any class). PNGs are RGB-rendered
      colormap spectrograms; we use the luminance channel as a scalar
      surrogate.
    - ``lab_131204``: ``(sample_size, *target_shape)`` STFT slices from
      WAVs in ``lab_wav_dir``, resampled 300→250 kHz.
    - ``wild_5970``: same as ``lab_131204`` but from ``wild_wav_dir``.

    Raises ``FileNotFoundError`` / ``RuntimeError`` if any cohort has
    insufficient usable input. The cleaning pipeline is shape-uniform
    across cohorts after this loader, which is required by ``knn_*``,
    ``raw_pixel_pca_d``, and the diagnostic VAE.
    """
    rng = np.random.default_rng(seed)

    # --- VocalMat: scan PNGs, filter to ones that actually exist on disk
    vocalmat_pngs = sorted(vocalmat_dir.rglob("*.png"))
    if not vocalmat_pngs:
        raise FileNotFoundError(
            f"VocalMat directory {vocalmat_dir} has no PNG files. Run "
            "scripts/cnn_download_vocalmat_sample.py first."
        )
    chosen_pngs = list(rng.choice(np.array(vocalmat_pngs, dtype=object),
                                  size=min(sample_size, len(vocalmat_pngs)),
                                  replace=False))
    if len(chosen_pngs) < sample_size:
        raise RuntimeError(
            f"VocalMat sample dir has only {len(chosen_pngs)} PNGs; "
            f"requested {sample_size}. Re-run the download script."
        )
    vocalmat = np.stack(
        [_png_to_luminance(p, target_shape) for p in chosen_pngs], axis=0,
    ).astype(np.float32)

    # --- Lab + wild: gather WAV paths. Recursive so directories that
    # nest recordings under per-session subfolders still work; flat
    # directories (current `USV_lab_131204/` and `5970 USV/`) are also fine.
    lab_wavs = sorted(lab_wav_dir.rglob("*.wav"))
    wild_wavs = sorted(wild_wav_dir.rglob("*.wav"))
    if not lab_wavs:
        raise FileNotFoundError(f"No .wav in {lab_wav_dir}")
    if not wild_wavs:
        raise FileNotFoundError(f"No .wav in {wild_wav_dir}")

    lab_specs = _wav_to_spectrograms(
        lab_wavs, n_windows=sample_size,
        target_sr=target_sample_rate_hz, target_shape=target_shape,
        window_seconds=_REAL_WINDOW_DURATION_S,
        n_fft=STFT_N_FFT, hop=STFT_HOP, rng=rng,
    )
    wild_specs = _wav_to_spectrograms(
        wild_wavs, n_windows=sample_size,
        target_sr=target_sample_rate_hz, target_shape=target_shape,
        window_seconds=_REAL_WINDOW_DURATION_S,
        n_fft=STFT_N_FFT, hop=STFT_HOP, rng=rng,
    )

    return {
        "vocalmat": vocalmat,
        "lab_131204": lab_specs,
        "wild_5970": wild_specs,
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
        # Real-data path (Module 18.2a): VocalMat PNGs from `vocalmat-sample`
        # dir, lab + wild STFTs computed from WAVs in their respective
        # `--*-sample` directories. All three cohorts standardised to 227×227
        # float32 spectrograms so K-NN / PCA / VAE can run uniformly.
        print(
            f"[info] Real-data run: vocalmat={args.vocalmat_sample}, "
            f"lab={args.lab_131204_sample}, wild={args.wild_5970_sample}, "
            f"sample_size={args.sample_size}",
            file=sys.stderr,
            flush=True,
        )
        t_load = time.monotonic()
        cohort_specs = _load_real_cohorts(
            vocalmat_dir=args.vocalmat_sample,
            lab_wav_dir=args.lab_131204_sample,
            wild_wav_dir=args.wild_5970_sample,
            sample_size=args.sample_size,
        )
        print(
            f"[info] Real-data load complete in "
            f"{time.monotonic() - t_load:.1f}s — cohort shapes: "
            + ", ".join(f"{cid}={v.shape}" for cid, v in cohort_specs.items()),
            file=sys.stderr,
            flush=True,
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
