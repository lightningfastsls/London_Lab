"""Render "v1-faithful" UNMASKED spectrogram patches for the lab_classifier_v1 oracle.

GOAL
====
Reproduce, BYTE-FOR-BYTE, the v1 TRAINING/held-out patch render so the
``results/lab_classifier_v1/best.pt`` oracle sees in-distribution inputs. A
prior attempt fed Stack-4 contour-MASKED patches and the oracle collapsed to
~75% Noise (masked patches are out-of-distribution). This render instead
replays ``archive/.../prepare_held_out_844.py::_extract_patch`` exactly: same
resample, same ``_spectrogram_db``, same ``clean_spectrogram(cfg, recording_id)``
on the WHOLE 2 s chunk (Constraint C2: global, never per-window), same crop
window (0.22 s centred on the detection midpoint), same ``_spec_to_uint8_patch``
(227x227 RGB).

NO MATH IS REWRITTEN HERE. The canonical building blocks are IMPORTED:
  - ``_spectrogram_db``, ``_spec_to_uint8_patch``, ``_PATCH_SIDE``,
    ``_VOCALMAT_STFT_HOP`` from the archived
    ``cnn_prepare_training_data`` (Stack-1).
  - ``CleaningConfig``, ``clean_spectrogram`` from the archived
    ``classifier.cleaning_pipeline`` (Stack-1). NOTE: these used to live in
    ``usv_spectrogram.classifier`` (where prepare_held_out_844 imports them
    from), but the Stack-1 archival (commit 67deb0c5) moved them into
    ``archive/cleaning_legacy/stack1/src/classifier/``. We import the archived
    copies so the render is identical to the one that produced held_out_844.
  - ``resample_to_vocalmat``, ``SOURCE_SAMPLE_RATE_HZ``, ``TARGET_SAMPLE_RATE_HZ``
    from the LIVE ``usv_spectrogram.classifier`` (unchanged by the archival).

The ONLY new code is iterating a detection CSV (or a held-out manifest, for the
pixel-fidelity validation) instead of the four cluster-review CSVs.

Usage (real lab render)::

    .venv/bin/python scripts/experiments/render_v1_faithful_patches.py \\
        --csv classified_detections_lab_131204_clean.csv \\
        --wav-root USV_lab_131204_chunked_2s_full/ \\
        --out-dir data/alpha3_oracle_patches/ \\
        --workers 4 --limit 0

Usage (844 pixel-fidelity validation, manifest mode)::

    .venv/bin/python scripts/experiments/render_v1_faithful_patches.py \\
        --manifest data/lab_cnn_training/held_out_844/manifest.csv \\
        --wav-root USV_lab_131204_chunked_2s_full/ \\
        --out-dir /tmp/v1_faithful_validation/ \\
        --workers 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- Path bootstrap --------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = REPO_ROOT / "src"
_ARCHIVE_STACK1_SRC = REPO_ROOT / "archive" / "cleaning_legacy" / "stack1" / "src"
_ARCHIVE_STACK1_SCRIPTS = REPO_ROOT / "archive" / "cleaning_legacy" / "stack1" / "scripts"
for _p in (_SRC, _ARCHIVE_STACK1_SRC, _ARCHIVE_STACK1_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# IMPORT (do NOT copy) the canonical training-render building blocks.
#
# The archived ``cnn_prepare_training_data`` does, at module-import time,
# ``from usv_spectrogram.classifier import CleaningConfig, clean_spectrogram``.
# The Stack-1 archival (commit 67deb0c5) removed those two names from the LIVE
# ``usv_spectrogram.classifier`` package and moved their definitions to
# ``archive/cleaning_legacy/stack1/src/classifier/cleaning_pipeline.py``. To let
# the archived script import unchanged — so its ``_spectrogram_db`` /
# ``_spec_to_uint8_patch`` are the byte-identical originals — we re-attach the
# ARCHIVED cleaning symbols onto the live package namespace BEFORE importing it.
# This restores exactly the import that existed when held_out_844 was rendered;
# no math is altered. The 844-pixel validation below is the proof of fidelity.
import usv_spectrogram.classifier as _live_classifier  # noqa: E402
from classifier.cleaning_pipeline import (  # noqa: E402  (archived Stack-1 copy)
    CleaningConfig,
    clean_spectrogram,
)
if not hasattr(_live_classifier, "CleaningConfig"):
    _live_classifier.CleaningConfig = CleaningConfig  # type: ignore[attr-defined]
if not hasattr(_live_classifier, "clean_spectrogram"):
    _live_classifier.clean_spectrogram = clean_spectrogram  # type: ignore[attr-defined]

from cnn_prepare_training_data import (  # noqa: E402
    _PATCH_SIDE,
    _VOCALMAT_STFT_HOP,
    _spec_to_uint8_patch,
    _spectrogram_db,
)
from usv_spectrogram.classifier import (  # noqa: E402
    TARGET_SAMPLE_RATE_HZ,
)
from usv_spectrogram.classifier.resample import (  # noqa: E402
    SOURCE_SAMPLE_RATE_HZ,
    resample_to_vocalmat,
)

# Patch window duration locked to the v1 training/held-out grid (ROADMAP D1).
_PATCH_DURATION_S: float = 0.22


# ---------------------------------------------------------------------------
# Canonical patch extraction — a VERBATIM replay of
# ``prepare_held_out_844._extract_patch`` (only the docstring is ours).
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Input normalization: real lab CSV OR held-out manifest -> common schema.
# Output schema columns: wav_stem, det_start_s, det_end_s, det_index.
# ---------------------------------------------------------------------------
def _load_jobs(args: argparse.Namespace) -> pd.DataFrame:
    if args.manifest is not None:
        df = pd.read_csv(args.manifest)
        for col in ("wav_stem", "det_start_s", "det_end_s"):
            if col not in df.columns:
                raise SystemExit(f"ERROR: --manifest missing required column '{col}'")
        # held_out_844 manifest has no det_index. Assign a per-(wav_stem) running
        # index so call_ids are unique; this is ONLY used to name the validation
        # output files and never affects pixels.
        if "det_index" not in df.columns:
            df = df.copy()
            df["det_index"] = df.groupby("wav_stem").cumcount()
        return df[["wav_stem", "det_start_s", "det_end_s", "det_index"]].copy()

    df = pd.read_csv(args.csv)
    for col in ("wav_stem", "det_start_s", "det_end_s", "det_index"):
        if col not in df.columns:
            raise SystemExit(f"ERROR: --csv missing required column '{col}'")
    return df[["wav_stem", "det_start_s", "det_end_s", "det_index"]].copy()


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------
def _render_one(task: dict) -> dict:
    """Render a single call. Returns a manifest-row dict (status in 'status')."""
    from PIL import Image

    wav_path = Path(task["wav_path"])
    out_path = Path(task["out_path"])
    cfg = CleaningConfig(baseline_mode="percentile")  # v1 watch-out #1 (MANDATORY)

    if out_path.exists() and not task["overwrite"]:
        return {
            "call_id": task["call_id"], "path": str(out_path),
            "wav_stem": task["wav_stem"], "status": "skipped_exists",
        }
    try:
        rgb, is_zero = _extract_patch(
            wav_path, task["det_start_s"], task["det_end_s"], _PATCH_DURATION_S, cfg
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "call_id": task["call_id"], "path": "",
            "wav_stem": task["wav_stem"], "status": f"error:{type(exc).__name__}:{exc}",
        }
    Image.fromarray(rgb, mode="RGB").save(out_path)
    return {
        "call_id": task["call_id"], "path": str(out_path),
        "wav_stem": task["wav_stem"],
        "status": "all_zero" if is_zero else "ok",
    }


# ---------------------------------------------------------------------------
# WAV resolution (recursive rglob, v1 watch-out #2)
# ---------------------------------------------------------------------------
def _resolve_wav(wav_root: Path, stem: str, cache: dict[str, Path | None]) -> Path | None:
    if stem in cache:
        return cache[stem]
    direct = wav_root / f"{stem}.wav"
    if direct.is_file():
        cache[stem] = direct
        return direct
    hits = list(wav_root.rglob(f"{stem}.wav"))
    cache[stem] = hits[0] if hits else None
    return cache[stem]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path,
                     help="Detection CSV (cols: wav_stem, det_start_s, det_end_s, det_index).")
    src.add_argument("--manifest", type=Path,
                     help="held_out_844-style manifest (cols: wav_stem, det_start_s, det_end_s); "
                          "det_index auto-assigned. Used for pixel-fidelity validation.")
    p.add_argument("--wav-root", type=Path, default=Path("USV_lab_131204_chunked_2s_full"),
                   help="Root dir of the 2 s lab WAV chunks (searched recursively).")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="0 = render all rows.")
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    import multiprocessing as mp

    args = _build_parser().parse_args(argv)
    cfg = CleaningConfig(baseline_mode="percentile")

    print("=" * 72)
    print("render_v1_faithful_patches — UNMASKED v1-faithful oracle substrate")
    print("=" * 72)
    print("PARAMETERS")
    print(f"  mode               : {'manifest (validation)' if args.manifest else 'csv (lab render)'}")
    print(f"  csv                : {args.csv}")
    print(f"  manifest           : {args.manifest}")
    print(f"  wav-root           : {args.wav_root}")
    print(f"  out-dir            : {args.out_dir}")
    print(f"  workers            : {args.workers}")
    print(f"  limit              : {args.limit} (0 = all)")
    print(f"  overwrite          : {args.overwrite}")
    print(f"  patch-side         : {_PATCH_SIDE}x{_PATCH_SIDE} (RGB)")
    print(f"  patch-duration-s   : {_PATCH_DURATION_S} (centred on detection midpoint)")
    print(f"  STFT hop           : {_VOCALMAT_STFT_HOP} (250 kHz VocalMat grid)")
    print(f"  resample           : {SOURCE_SAMPLE_RATE_HZ} -> {TARGET_SAMPLE_RATE_HZ} Hz")
    print(f"  cleaning baseline  : {cfg.baseline_mode} (watch-out #1, MANDATORY)")
    print(f"  cleaning cfg       : {cfg}")
    print("-" * 72)

    df = _load_jobs(args)
    print(f"INPUT rows (raw)     : {len(df)}")
    df = df.drop_duplicates(subset=["wav_stem", "det_start_s", "det_end_s"]).reset_index(drop=True)
    print(f"INPUT rows (deduped) : {len(df)}")
    if args.limit and args.limit > 0:
        df = df.head(args.limit).reset_index(drop=True)
        print(f"INPUT rows (limited) : {len(df)}")
    print("-" * 72)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Build tasks; resolve WAVs up front (sequential rglob, cached).
    wav_cache: dict[str, Path | None] = {}
    tasks: list[dict] = []
    n_missing = 0
    seen_ids: set[str] = set()
    for _, row in df.iterrows():
        stem = str(row["wav_stem"])
        wav_path = _resolve_wav(args.wav_root, stem, wav_cache)
        if wav_path is None:
            n_missing += 1
            if n_missing <= 10:
                print(f"  MISSING WAV: {stem}", file=sys.stderr)
            continue
        call_id = f"{stem}__det{int(row['det_index'])}"
        if call_id in seen_ids:
            call_id = f"{call_id}_{len(tasks)}"
        seen_ids.add(call_id)
        tasks.append({
            "call_id": call_id,
            "wav_stem": stem,
            "wav_path": str(wav_path),
            "out_path": str(args.out_dir / f"{call_id}.png"),
            "det_start_s": float(row["det_start_s"]),
            "det_end_s": float(row["det_end_s"]),
            "overwrite": args.overwrite,
        })

    print(f"  resolvable jobs    : {len(tasks)}")
    print(f"  missing WAVs       : {n_missing}")
    print("-" * 72)

    if args.workers > 1 and len(tasks) > 1:
        with mp.Pool(processes=args.workers) as pool:
            results = pool.map(_render_one, tasks)
    else:
        results = [_render_one(t) for t in tasks]

    # Status tally
    status_counts: dict[str, int] = {}
    errors: list[str] = []
    for r in results:
        s = r["status"]
        key = s if not s.startswith("error:") else "error"
        status_counts[key] = status_counts.get(key, 0) + 1
        if s.startswith("error:") and len(errors) < 10:
            errors.append(f"{r['call_id']}: {s}")

    manifest = pd.DataFrame(
        [{"call_id": r["call_id"], "path": r["path"], "wav_stem": r["wav_stem"]}
         for r in results if r["status"] in ("ok", "all_zero", "skipped_exists")]
    )
    manifest_path = args.out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print("RESULTS")
    print(f"  status counts      : {status_counts}")
    print(f"  manifest rows      : {len(manifest)}")
    print(f"  manifest           : {manifest_path}")
    print(f"  out-dir            : {args.out_dir}")
    if errors:
        print("  first errors:")
        for e in errors:
            print(f"    {e}", file=sys.stderr)
    n_zero = status_counts.get("all_zero", 0)
    if n_zero > 0:
        print(f"WARNING: {n_zero} all-zero patches — cleaning watch-out #1 may have "
              "regressed; inspect before trusting the oracle.", file=sys.stderr)
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
