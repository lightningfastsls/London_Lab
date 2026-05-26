"""Module 18.4 Tier-2 prerequisite: re-extract the 844 held-out lab patches.

The four cluster/noise review CSVs in ``results/lab_{cluster0,cluster1,cluster2,
noise}_review/review_index_annotated.csv`` carry human ``usv``/``noise`` verdicts
for 844 lab detections. The annotated review figures in those directories are
variable-size RGBA visualisations — they do NOT join to the model patch grid.
This script re-extracts clean 227x227 model patches for those 844 detections so
the DANN (and 18.3 baseline) classifier can be evaluated on real lab ground truth.

Design — match the TRAINING preprocessing exactly, then crop the detection window
=================================================================================
The 227k unlabeled "lab" training patches were produced by ``cnn_prepare_training
_data._wav_to_patches`` which BLINDLY TILES the whole recording into consecutive
0.22 s patches. The held-out set instead needs a patch CENTERED on each
detection's ``(det_start_s, det_end_s)`` so it can carry the usv/noise verdict.

To keep eval patches geometrically identical to training patches we reuse the
same building blocks from ``cnn_prepare_training_data``:

  resample_to_vocalmat (300k -> 250k)  ->  _spectrogram_db (Hamming-256 / hop-128
  / NFFT-1024 @ 250 kHz, the locked VocalMat CNN training-grid)  ->  clean_spectro
  gram(cfg=percentile)  ->  crop 0.22 s window centered on detection midpoint  ->
  _spec_to_uint8_patch (227x227 RGB).

Constraint C2 (global-MAD on the WHOLE spectrogram, THEN crop) is preserved:
``clean_spectrogram`` runs on the full 2 s chunk spectrogram before we crop.

Watch-outs inherited from the 18.2b/18.3 Tier-2 chain
=====================================================
1. ``CleaningConfig(baseline_mode="percentile")`` is MANDATORY, not preferred:
   ``median_envelope`` chained with global-MAD floors most cells and emits
   all-black patches (2,734 witnessed 2026-05-22). We reuse ``percentile``.
2. WAV stems must be resolved with ``rglob`` (recursive) — the original
   ``_collect_wav_rows`` used non-recursive ``glob`` and silently skipped
   nested WAVs.

Output
======
  data/lab_cnn_training/held_out_844/
    manifest.csv          # columns: path, usv_verdict, + metadata
    patches/*.png         # one 227x227 RGB patch per resolved detection

The manifest's ``path`` + ``usv_verdict`` columns are what the held-out evaluator
consumes. NOTE: ``training._evaluate_held_out_845`` is currently a placeholder
(label-distribution proxy); the real patch-loading inference path is still TODO
for the DANN eval step.

Usage::

    .venv/bin/python scripts/prepare_held_out_844.py \\
        --review-dir results/ \\
        --wav-root USV_lab_131204_chunked_2s_full/ \\
        --output-dir data/lab_cnn_training/held_out_844/ \\
        --patch-duration-s 0.22
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Path bootstrap: this script lives in scripts/; the package is at src/.
REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Reuse the EXACT training preprocessing building blocks so held-out patches
# share the locked CNN training-grid geometry with the training patches.
from cnn_prepare_training_data import (  # noqa: E402
    _PATCH_SIDE,
    _VOCALMAT_STFT_HOP,
    _spec_to_uint8_patch,
    _spectrogram_db,
)
from usv_spectrogram.classifier import (  # noqa: E402
    TARGET_SAMPLE_RATE_HZ,
    CleaningConfig,
    clean_spectrogram,
)
from usv_spectrogram.classifier.resample import (  # noqa: E402
    SOURCE_SAMPLE_RATE_HZ,
    resample_to_vocalmat,
)

# The four review CSVs that carry the 844 verdicts (relative to --review-dir).
_REVIEW_CSVS: tuple[str, ...] = (
    "lab_cluster0_review/review_index_annotated.csv",
    "lab_cluster1_review/review_index_annotated.csv",
    "lab_cluster2_review/review_index_annotated.csv",
    "lab_noise_review/review_index_annotated.csv",
)


def _load_review_rows(review_dir: Path) -> pd.DataFrame:
    """Concatenate the four review CSVs, tagging each row with its source cluster."""
    frames: list[pd.DataFrame] = []
    for rel in _REVIEW_CSVS:
        csv_path = review_dir / rel
        if not csv_path.is_file():
            print(f"  WARNING: missing review CSV: {csv_path}", file=sys.stderr)
            continue
        df = pd.read_csv(csv_path)
        df["source_review"] = rel.split("/")[0]
        frames.append(df)
        print(f"  [{rel.split('/')[0]:20s}] {len(df):4d} rows")
    if not frames:
        raise SystemExit("ERROR: no review CSVs found under --review-dir")
    return pd.concat(frames, ignore_index=True)


def _resolve_wav(wav_root: Path, stem: str, cache: dict[str, Path | None]) -> Path | None:
    """Resolve a wav_stem to a path via recursive rglob (watch-out #2)."""
    if stem in cache:
        return cache[stem]
    hits = list(wav_root.rglob(f"{stem}.wav"))
    cache[stem] = hits[0] if hits else None
    return cache[stem]


def _extract_patch(
    wav_path: Path,
    det_start_s: float,
    det_end_s: float,
    patch_duration_s: float,
    cfg: CleaningConfig,
) -> tuple[np.ndarray, bool]:
    """Return (227x227x3 uint8 patch, is_all_zero) centered on the detection midpoint."""
    import soundfile as sf

    samples, sr_in = sf.read(str(wav_path), dtype="float32")
    if samples.ndim > 1:
        samples = samples[:, 0]

    if sr_in == SOURCE_SAMPLE_RATE_HZ:
        samples_250 = resample_to_vocalmat(samples)
        sr_eff = TARGET_SAMPLE_RATE_HZ
    elif sr_in == TARGET_SAMPLE_RATE_HZ:
        samples_250 = samples.astype(np.float32, copy=False)
        sr_eff = TARGET_SAMPLE_RATE_HZ
    else:
        samples_250 = samples.astype(np.float32, copy=False)
        sr_eff = int(sr_in)

    # C2: clean the WHOLE chunk spectrogram, THEN crop.
    spec_db = _spectrogram_db(samples_250, sr_eff)
    cleaned = clean_spectrogram(spec_db, cfg, recording_id=wav_path.stem)
    n_time = cleaned.shape[1]

    frames_per_patch = max(1, int(round(patch_duration_s * sr_eff / _VOCALMAT_STFT_HOP)))
    midpoint_s = 0.5 * (float(det_start_s) + float(det_end_s))
    center_f = int(round(midpoint_s * sr_eff / _VOCALMAT_STFT_HOP))
    half = frames_per_patch // 2
    start_f = max(0, center_f - half)
    end_f = min(n_time, start_f + frames_per_patch)
    start_f = max(0, end_f - frames_per_patch)  # re-anchor if clamped at the right edge

    slab = cleaned[:, start_f:end_f]
    patch_gray = _spec_to_uint8_patch(slab)
    is_all_zero = bool(patch_gray.max() == 0)
    rgb = np.stack([patch_gray, patch_gray, patch_gray], axis=-1)
    return rgb, is_all_zero


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--review-dir", type=Path, default=Path("results"),
                   help="Directory containing lab_{cluster0,cluster1,cluster2,noise}_review/")
    p.add_argument("--wav-root", type=Path, default=Path("USV_lab_131204_chunked_2s_full"),
                   help="Root dir of the 2 s lab WAV chunks (searched recursively).")
    p.add_argument("--output-dir", type=Path,
                   default=Path("data/lab_cnn_training/held_out_844"))
    p.add_argument("--patch-duration-s", type=float, default=0.22,
                   help="ROADMAP D1: 0.22 s primary (must match training patches).")
    return p


def main(argv: list[str] | None = None) -> int:
    from PIL import Image

    args = _build_parser().parse_args(argv)
    cfg = CleaningConfig(baseline_mode="percentile")  # watch-out #1

    print("=" * 70)
    print("Module 18.4 Tier-2 prep — re-extract 844 held-out lab patches")
    print("=" * 70)
    print("PARAMETERS")
    print(f"  review-dir         : {args.review_dir}")
    print(f"  wav-root           : {args.wav_root}")
    print(f"  output-dir         : {args.output_dir}")
    print(f"  patch-duration-s   : {args.patch_duration_s}")
    print(f"  patch-side         : {_PATCH_SIDE}x{_PATCH_SIDE} (RGB)")
    print(f"  STFT               : Hamming win/hop/nfft via cnn_prepare (250 kHz grid)")
    print(f"  resample           : {SOURCE_SAMPLE_RATE_HZ} -> {TARGET_SAMPLE_RATE_HZ} Hz")
    print(f"  cleaning baseline  : {cfg.baseline_mode} (watch-out #1)")
    print(f"  window strategy    : 0.22 s centered on detection midpoint")
    print("-" * 70)

    print("INPUT review CSVs:")
    df = _load_review_rows(args.review_dir)
    print(f"  TOTAL rows         : {len(df)}")

    # Required columns (verified against the real CSV header, not the handoff prose).
    for col in ("wav_stem", "det_start_s", "det_end_s", "verdict"):
        if col not in df.columns:
            raise SystemExit(f"ERROR: review CSV missing required column '{col}'")

    # Dedupe defensively (clusters should be disjoint).
    before = len(df)
    df = df.drop_duplicates(subset=["wav_stem", "det_start_s", "det_end_s"]).reset_index(drop=True)
    if len(df) != before:
        print(f"  deduped            : {before} -> {len(df)} (removed {before - len(df)})")

    verdict_counts = df["verdict"].astype(str).str.lower().value_counts()
    print(f"  verdict breakdown  : {verdict_counts.to_dict()}")
    print("-" * 70)

    patches_dir = args.output_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    wav_cache: dict[str, Path | None] = {}
    manifest_rows: list[dict] = []
    n_missing = 0
    n_all_zero = 0
    seen_names: set[str] = set()

    for i, row in df.iterrows():
        stem = str(row["wav_stem"])
        wav_path = _resolve_wav(args.wav_root, stem, wav_cache)
        if wav_path is None:
            n_missing += 1
            if n_missing <= 10:
                print(f"  MISSING WAV: {stem}", file=sys.stderr)
            continue

        rgb, is_zero = _extract_patch(
            wav_path, row["det_start_s"], row["det_end_s"], args.patch_duration_s, cfg
        )
        if is_zero:
            n_all_zero += 1

        start_ms = int(round(float(row["det_start_s"]) * 1000))
        name = f"{stem}_{start_ms:04d}ms.png"
        if name in seen_names:  # guarantee uniqueness on the rare start collision
            name = f"{stem}_{start_ms:04d}ms_{i}.png"
        seen_names.add(name)
        patch_path = patches_dir / name
        Image.fromarray(rgb, mode="RGB").save(patch_path)

        manifest_rows.append({
            "path": str(patch_path.relative_to(REPO_ROOT)) if patch_path.is_absolute()
                    else str(patch_path),
            "usv_verdict": str(row["verdict"]).strip().lower(),
            "wav_stem": stem,
            "det_start_s": float(row["det_start_s"]),
            "det_end_s": float(row["det_end_s"]),
            "source_review": row.get("source_review", ""),
            "traditional_label": row.get("traditional_label", ""),
            "couple": row.get("couple", ""),
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = args.output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print("-" * 70)
    print("RESULTS")
    print(f"  patches written    : {len(manifest_rows)}")
    print(f"  missing WAVs       : {n_missing}")
    print(f"  all-zero patches   : {n_all_zero}  (>0 indicates a cleaning regression!)")
    if len(manifest):
        print(f"  manifest verdicts  : {manifest['usv_verdict'].value_counts().to_dict()}")
    print(f"  manifest           : {manifest_path}")
    print(f"  patches dir        : {patches_dir}")
    print("=" * 70)

    if n_all_zero > 0:
        print("WARNING: all-zero patches detected — cleaning watch-out #1 may have "
              "regressed; inspect before trusting eval.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
